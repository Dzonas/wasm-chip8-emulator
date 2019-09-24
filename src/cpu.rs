mod instructions;

use instructions::INSTRUCTION_TABLE;
use std::error::Error;
use std::fmt;
use wasm_bindgen::prelude::*;
use super::utils;

const MEMORY_SIZE: usize = 4096;
const N_REGISTERS: usize = 16;
const STACK_SIZE: usize = 16;
pub const SCREEN_WIDTH: usize = 64;
pub const SCREEN_HEIGHT: usize = 32;
pub const KEYBOARD_SIZE: usize = 16;
const PROGRAM_START: usize = 0x200;
const FONTS_START: usize = 0x000;
const FONTS: [u8; 5 * 16] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70, 0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0,
    0x10, 0xF0, 0x10, 0xF0, 0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0, 0xF0, 0x80,
    0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40, 0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0,
    0x10, 0xF0, 0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0, 0xF0, 0x80, 0x80, 0x80,
    0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0, 0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80,
];

#[derive(Debug)]
pub enum CPUError {
    OutOfBounds,    // Tried to read out of bounds memory
    StackUnderflow, // Tried to pop on an empty stack
    StackOverflow,  // Tried to push on a full stack
    OutOfMemory,
    UnknownOpcode,
    UnknownDigit,
    UnknownKey,
}

impl CPUError {
    fn desc(&self) -> &'static str {
        match *self {
            CPUError::OutOfBounds => "tried to read memory out of bounds",
            CPUError::StackUnderflow => "tried to pop on an empty stack",
            CPUError::StackOverflow => "tried to push on a full stack",
            CPUError::OutOfMemory => "out of memory",
            CPUError::UnknownOpcode => "tried to execute unknown opcode",
            CPUError::UnknownDigit => "tried to use unknown digit from font table",
            CPUError::UnknownKey => "tried to set an unknown key"
        }
    }
}

impl Error for CPUError {}

impl fmt::Display for CPUError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.desc())
    }
}

#[wasm_bindgen]
pub struct Chip8 {
    memory: Vec<u8>,
    registers: Vec<u8>,
    index_register: u16,
    program_counter: u16,
    stack_pointer: u8,
    stack: Vec<u16>,
    delay_timer: u8,
    sound_timer: u8,
    screen_buffer: Vec<u8>,
    keyboard: Vec<bool>,
    load_store_quirk: bool, // 0xFX55/0xFX65 don't touch the index register
    shift_quirk: bool,      // 0x8XY6/0x8XYE shift Vx instead of Vy
    wrapping: bool, // Should DRW instruction wrap sprites around
    waiting_for_key: bool,
    waiting_for_key_reg: u8,
}

#[wasm_bindgen]
impl Chip8 {
    pub fn new() -> Chip8 {
        utils::set_panic_hook();

        let mut memory = vec![0; MEMORY_SIZE];
        let registers = vec![0; N_REGISTERS];
        let index_register = 0;
        let program_counter = PROGRAM_START as u16;
        let stack_pointer = 0;
        let stack = vec![0; STACK_SIZE];
        let delay_timer = 0;
        let sound_timer = 0;
        let screen_buffer = vec![0; SCREEN_WIDTH * SCREEN_HEIGHT];
        let keyboard = vec![false; KEYBOARD_SIZE];
        let load_store_quirk = false;
        let shift_quirk = false;
        let waiting_for_key = false;
        let waiting_for_key_reg = 0;
        let wrapping = true;

        memory[FONTS_START..FONTS.len()].clone_from_slice(&FONTS[..]);

        Chip8 {
            memory,
            registers,
            index_register,
            program_counter,
            stack_pointer,
            stack,
            delay_timer,
            sound_timer,
            screen_buffer,
            keyboard,
            load_store_quirk,
            shift_quirk,
            wrapping,
            waiting_for_key,
            waiting_for_key_reg,
        }
    }

    pub fn reset(&mut self) {
        self.memory = vec![0; MEMORY_SIZE];
        self.registers = vec![0; N_REGISTERS];
        self.index_register = 0;
        self.program_counter = PROGRAM_START as u16;
        self.stack_pointer = 0;
        self.stack = vec![0; STACK_SIZE];
        self.delay_timer = 0;
        self.sound_timer = 0;
        self.screen_buffer = vec![0; SCREEN_WIDTH * SCREEN_HEIGHT];
        self.keyboard = vec![false; KEYBOARD_SIZE];
        self.load_store_quirk = false;
        self.shift_quirk = false;
        self.wrapping = true;
        self.waiting_for_key = false;
        self.waiting_for_key_reg = 0;

        self.memory[FONTS_START..FONTS.len()].clone_from_slice(&FONTS[..]);
    }

    ///
    /// Fetches opcode from memory. Decodes opcode. Executes decoded instruction. Moves program
    /// counter to the next instruction.
    /// If 0xFx0A (wait for key) opcode was executed then cycle() does nothing until any key is
    /// pressed.
    ///
    pub fn cycle(&mut self) -> Result<(), JsValue> {
        if !self.waiting_for_key {
            let opcode = self.fetch_opcode().expect("holy");

            let id = ((opcode & 0xF000) >> 12) as usize;
            INSTRUCTION_TABLE[id](self, opcode).expect("mother");

            if id != 1 && id != 2 && (opcode & 0xF0FF) != 0xF00A {
                self.program_counter += 2;
            }

            if self.program_counter as usize >= self.memory.len() {
                return Err(JsValue::from_str(CPUError::OutOfMemory.description()));
            }
        } else {
            for i in 0..self.keyboard.len() {
                if self.keyboard[i] {
                    self.registers[self.waiting_for_key_reg as usize] = i as u8;
                    self.waiting_for_key = false;
                    self.program_counter += 2;
                    break;
                }
            }
        }

        Ok(())
    }

    ///
    /// Loads given program, starting at address "PROGRAM_START". If the program is larger than
    /// the memory from "PROGRAM_START" to the end - return an Err.
    ///
    pub fn load(&mut self, rom: &[u8]) -> Result<(), JsValue> {
        if PROGRAM_START + rom.len() >= self.memory.len() {
            return Err(JsValue::from_str(CPUError::OutOfBounds.description()));
        }

        self.memory[PROGRAM_START..PROGRAM_START + rom.len()].copy_from_slice(&rom[..]);

        Ok(())
    }

    ///
    /// Returns a pointer to a screen buffer. "1" - means pixel is on, "0" - pixel is off.
    ///
    pub fn screen_buffer(&self) -> *const u8 {
        self.screen_buffer.as_ptr()
    }

    ///
    /// Returns true if the sound should play. False otherwise.
    ///
    pub fn sound(&self) -> bool {
        self.sound_timer > 0
    }

    ///
    /// Decrements delay and sound timers. Should be executed at 60 Hz.
    ///
    pub fn update_timers(&mut self) {
        self.delay_timer = self.delay_timer.saturating_sub(1);
        self.sound_timer = self.sound_timer.saturating_sub(1);
    }

    ///
    /// Sets key "i" (from range 0x0 - 0xF) to pressed or unpressed. Returns Err if given key
    /// doesn't exist.
    ///
    pub fn set_key(&mut self, i: usize, value: bool) -> Result<(), JsValue> {
        let key = self.keyboard.get_mut(i);

        if let Some(k) = key {
            *k = value;
            Ok(())
        } else {
            Err(JsValue::from_str(CPUError::UnknownKey.description()))
        }
    }

    ///
    /// If set the CPU will use quirked behavior of 0xFX55 and 0xFX65 opcodes.
    ///
    pub fn set_load_store_quirk(&mut self, value: bool) {
        self.load_store_quirk = value;
    }

    ///
    /// If set the CPU will use quirked behavior of 0x8XY6 and 0x8XYE opcodes.
    ///
    pub fn set_shift_quirk(&mut self, value: bool) {
        self.shift_quirk = value
    }

    ///
    /// If set DRW instruction will wrap sprites around. Needed for compatibility issues.
    ///
    pub fn set_wrapping(&mut self, value: bool) { self.wrapping = value }
}

impl Chip8 {
    ///
    /// Returns opcode from memory address given by program counter.
    /// Joins two bytes to create one 2-byte opcode. Returns error if the program counter points
    /// outside of the memory or the program counter points at the odd address (opcodes begin at
    /// even addresses).
    ///
    fn fetch_opcode(&self) -> Result<u16, CPUError> {
        if (self.program_counter as usize) >= MEMORY_SIZE {
            return Err(CPUError::OutOfBounds);
        }

        let high = self.memory[self.program_counter as usize];
        let low = self.memory[self.program_counter as usize + 1];

        let opcode = ((u16::from(high)) << 8) | u16::from(low);

        Ok(opcode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    ///
    /// Helper function that adds opcode to location in memory given by instructions's program counter.
    ///
    fn set_opcode(cpu: &mut Chip8, opcode: u16) {
        cpu.memory[cpu.program_counter as usize] = ((opcode & 0xFF00) >> 8) as u8;
        cpu.memory[cpu.program_counter as usize + 1] = (opcode & 0x00FF) as u8;
    }

    #[test]
    fn test_fetch_opcode() {
        let mut chip8 = Chip8::new();
        let opcode = 0xA2F0;

        chip8.program_counter = 0x00;
        set_opcode(&mut chip8, opcode);

        // Test proper fetch
        let opcode = chip8.fetch_opcode().unwrap();
        assert_eq!(opcode, 0xA2F0);

        // Test out of bounds fetch
        chip8.program_counter = MEMORY_SIZE as u16;
        assert!(chip8.fetch_opcode().is_err());
    }

    #[test]
    fn test_wait_for_key() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF10A;
        let pc = chip8.program_counter;

        set_opcode(&mut chip8, opcode);
        assert!(chip8.cycle().is_ok());
        assert_eq!(chip8.program_counter, pc);

        chip8.keyboard[0xA] = true;
        assert!(chip8.cycle().is_ok());
        assert_eq!(chip8.registers[1], 0xA);
        assert_eq!(chip8.program_counter, pc + 2);
        assert!(!chip8.waiting_for_key);
    }
}
