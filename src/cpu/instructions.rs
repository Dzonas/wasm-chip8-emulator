use super::{CPUError, Chip8, SCREEN_WIDTH};
use crate::cpu::SCREEN_HEIGHT;
use js_sys::Math::random;

type CPUResult = Result<(), CPUError>;
type Instruction = dyn Fn(&mut Chip8, u16) -> CPUResult;

pub const INSTRUCTION_TABLE: [&Instruction; 16] = [
    &decode_0,
    &jp_addr,
    &call_addr,
    &se_vx_byte,
    &sne_vx_byte,
    &decode_5,
    &ld_vx_byte,
    &add_vx_byte,
    &decode_8,
    &decode_9,
    &ld_i_addr,
    &jp_v0_addr,
    &rnd_vx_byte,
    &decode_d,
    &decode_e,
    &decode_f,
];

///
/// Decodes instructions starting with 0x0.
///
fn decode_0(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    match opcode {
        0x0000 => Ok(()),
        0x00E0 => cls(cpu, opcode), // CLS
        0x00EE => ret(cpu, opcode), // RET
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Decodes instructions starting with 0x5.
///
fn decode_5(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let id = opcode & 0x000F;

    match id {
        0x0 => se_vx_vy(cpu, opcode),
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Decodes instructions starting with 0x8.
///
fn decode_8(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let id = opcode & 0x000F;

    match id {
        0x0 => ld_vx_vy(cpu, opcode),
        0x1 => or_vx_vy(cpu, opcode),
        0x2 => and_vx_vy(cpu, opcode),
        0x3 => xor_vx_vy(cpu, opcode),
        0x4 => add_vx_vy(cpu, opcode),
        0x5 => sub_vx_vy(cpu, opcode),
        0x6 => shr_vx_vy(cpu, opcode),
        0x7 => subn_vx_vy(cpu, opcode),
        0xE => shl_vx_vy(cpu, opcode),
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Decodes instructions starting with 0x9.
///
fn decode_9(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let id = opcode & 0x000F;

    match id {
        0x0 => sne_vx_vy(cpu, opcode),
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Decodes instructions starting with 0xD.
///
fn decode_d(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    if cpu.wrapping {
        drw_vx_vy_nibble(cpu, opcode)
    } else {
        drw_vx_vy_nibble_no_wrap(cpu, opcode)
    }
}

///
/// Decodes instructions starting with 0xE.
///
fn decode_e(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let id = opcode & 0x00FF;

    match id {
        0x9E => skp_vx(cpu, opcode),
        0xA1 => sknp_vx(cpu, opcode),
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Decodes instructions starting with 0xF.
///
fn decode_f(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let id = opcode & 0x00FF;

    match id {
        0x07 => ld_vx_dt(cpu, opcode),
        0x0A => ld_vx_k(cpu, opcode),
        0x15 => ld_dt_vx(cpu, opcode),
        0x18 => ld_st_vx(cpu, opcode),
        0x1E => add_i_vx(cpu, opcode),
        0x29 => ld_f_vx(cpu, opcode),
        0x33 => ld_b_vx(cpu, opcode),
        0x55 => ld_i_vx(cpu, opcode),
        0x65 => ld_vx_i(cpu, opcode),
        _ => Err(CPUError::UnknownOpcode),
    }
}

///
/// Clears display.
/// Sets all "pixels" to false.
///
fn cls(cpu: &mut Chip8, _: u16) -> CPUResult {
    cpu.screen_buffer.iter_mut().for_each(|v| *v = 0);
    Ok(())
}

///
/// Return from a subroutine. Sets program counter to value from the top of the stack and then
/// subtracts 1 from the stack pointer. Returns Err if the stack pointer was at 0 i.e. there
/// is nothing to return from.
///
fn ret(cpu: &mut Chip8, _: u16) -> CPUResult {
    if cpu.stack_pointer == 0 {
        return Err(CPUError::StackUnderflow);
    }

    cpu.program_counter = cpu.stack[cpu.stack_pointer as usize];
    cpu.stack_pointer -= 1;

    Ok(())
}

///
/// Jumps to address.
/// Sets program counter to the address value.
///
fn jp_addr(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let addr = opcode & 0x0FFF;
    cpu.program_counter = addr;

    Ok(())
}

///
/// Call a subroutine at addr.
/// Increments stack pointer, then puts program counter at the top of the stack. After that
/// the program counter is set to the addr. Returns error if the stack is already full.
///
fn call_addr(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let addr = opcode & 0x0FFF;
    if cpu.stack_pointer as usize >= cpu.stack.len() - 1 {
        return Err(CPUError::StackOverflow);
    }

    cpu.stack_pointer += 1;
    cpu.stack[cpu.stack_pointer as usize] = cpu.program_counter;
    cpu.program_counter = addr;

    Ok(())
}

///
/// Skips next instruction if Vx == byte.
/// I
///
fn se_vx_byte(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let byte = (opcode & 0x00FF) as u8;

    if cpu.registers[x] == byte {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Skips next instruction if Vx != byte.
/// If that's the case increments program counter by 2.
///
fn sne_vx_byte(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let byte = (opcode & 0x00FF) as u8;

    if cpu.registers[x] != byte {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Skips next instruction if Vx == Vy.
///
fn se_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    if cpu.registers[x] == cpu.registers[y] {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Loads byte to Vx register.
///
fn ld_vx_byte(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let byte = (opcode & 0x00FF) as u8;

    cpu.registers[x] = byte;

    Ok(())
}

///
/// Adds byte to Vx register. Result is loaded into Vx register.
///
fn add_vx_byte(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let byte = (opcode & 0x00FF) as u8;

    cpu.registers[x] = cpu.registers[x].wrapping_add(byte);

    Ok(())
}

///
/// Loads byte from Vy register to Vx register.
///
fn ld_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    cpu.registers[x] = cpu.registers[y];

    Ok(())
}

///
/// Performs bitwise or on values in Vx and Vy registers
/// and then result is loaded into Vx register.
///
fn or_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    cpu.registers[x] |= cpu.registers[y];

    Ok(())
}

///
/// Performs bitwise and on values in Vx and Vy registers
/// and then result is loaded into Vx register.
///
fn and_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    cpu.registers[x] &= cpu.registers[y];

    Ok(())
}

///
/// Performs bitwise xor on values in Vx and Vy registers
/// and then result is loaded into Vx register.
///
fn xor_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    cpu.registers[x] ^= cpu.registers[y];

    Ok(())
}

///
/// Adds values inside Vx and Vy registers. Result is loaded into Vx register. If the operation
/// overflows carry flag is set to 1 inside VF register and only least significant 8-bits are
/// considered.
///
fn add_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    let (result, carry_flag) = cpu.registers[x].overflowing_add(cpu.registers[y]);

    cpu.registers[x] = result;
    cpu.registers[15] = carry_flag as u8;

    Ok(())
}

///
/// Subtracts Vy from Vx and stores it in Vx. VF is set to "not borrow".
///
fn sub_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    let (v, cf) = cpu.registers[x].overflowing_sub(cpu.registers[y]);

    cpu.registers[x] = v;
    cpu.registers[0xF] = !cf as u8;

    Ok(())
}

///
/// Stores Vy, shifted right by one bit, in Vx. Loads least significant bit from the Vy, before
/// the shift, into VF.
///
fn shr_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    if cpu.shift_quirk {
        // Quirked behavior
        cpu.registers[0xF] = cpu.registers[x] & 0b1;
        cpu.registers[x] >>= 1;
    } else {
        // Original behavior
        cpu.registers[0xF] = cpu.registers[y] & 0b1;
        cpu.registers[x] = cpu.registers[y] >> 1;
    }

    Ok(())
}

///
/// Subtracts Vx from Vy and stores it in Vx. VF is set to "not borrow".
///
fn subn_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    let (v, cf) = cpu.registers[y].overflowing_sub(cpu.registers[x]);

    cpu.registers[x] = v;
    cpu.registers[0xF] = !cf as u8;

    Ok(())
}

///
/// Stores Vy, shifted left by one bit, in Vx. Loads most significant bit from the Vy, before
/// the shift, into VF.
///
fn shl_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    if cpu.shift_quirk {
        // Quirked behavior
        cpu.registers[0xF] = (cpu.registers[x] & 0x80) >> 7;
        cpu.registers[x] <<= 1;
    } else {
        // Original behavior
        cpu.registers[0xF] = (cpu.registers[y] & 0x80) >> 7;
        cpu.registers[x] = cpu.registers[y] << 1;
    }

    Ok(())
}

///
/// Skips next instruction if Vx != Vy.
/// If that's the case increments program counter by 2.
///
fn sne_vx_vy(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;

    if cpu.registers[x] != cpu.registers[y] {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Loads addr into I register.
///
fn ld_i_addr(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let addr = opcode & 0x0FFF;
    cpu.index_register = addr;

    Ok(())
}

///
/// Jumps to address V0 + addr.
/// Assumes that addr is a 4-bit value. Returns OutOfBounds error if v0 + addr is out of memory
/// bounds.
///
fn jp_v0_addr(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let addr = opcode & 0x0FFF;
    let new_pc = addr + u16::from(cpu.registers[0]);

    if new_pc as usize >= cpu.memory.len() {
        return Err(CPUError::OutOfBounds);
    }

    cpu.program_counter = new_pc;

    Ok(())
}

///
/// Generates random number ANDed with "byte" and loads it into Vx register.
///
fn rnd_vx_byte(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let byte = (opcode & 0x00FF) as u8;

    let r = (random() * f64::from(std::u8::MAX)) as u8;

    cpu.registers[x] = r & byte;

    Ok(())
}

///
/// Draws sprite at x = Vx, y = Vy coordinates. Sprite pixels are XORed onto the screen buffer.
/// If any of the pixels switched from 0 to 1, flag in VF register is set. Any pixel which
/// location is outside of the screen is wrapped back to the screen.
///
fn drw_vx_vy_nibble(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;
    let n = (opcode & 0x000F) as usize;

    let vx = cpu.registers[x] as usize;
    let vy = cpu.registers[y] as usize;
    let mut vf_flag = 0;

    for row in 0..n {
        let sprite_row = cpu.memory[cpu.index_register as usize + row];
        let ny = (vy + row) % SCREEN_HEIGHT;
        for col in 0..8 {
            let nx = (vx + col) % SCREEN_WIDTH;
            let pixel = (sprite_row >> (7 - col as u8)) & 1;
            let i = nx + ny * SCREEN_WIDTH;

            let prev_pixel = cpu.screen_buffer[i];
            let next_pixel = prev_pixel ^ pixel;

            // If pixel changed from 1 to 0 update flag in VF.
            if prev_pixel == 1 && next_pixel == 0 {
                vf_flag = 1;
            }

            cpu.screen_buffer[i] = next_pixel;
        }
    }

    cpu.registers[0xF] = vf_flag;

    Ok(())
}

///
/// Same as drw_vx_vy_nibble but doesn't wrap sprites around x and y axis.
///
fn drw_vx_vy_nibble_no_wrap(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let y = ((opcode & 0x00F0) >> 4) as usize;
    let n = (opcode & 0x000F) as usize;

    let vx = cpu.registers[x] as usize;
    let vy = cpu.registers[y] as usize;
    let mut vf_flag = 0;

    for row in 0..n {
        let sprite_row = cpu.memory[cpu.index_register as usize + row];
        let ny = vy + row;

        if ny >= SCREEN_HEIGHT {
            continue;
        }

        for col in 0..8 {
            let nx = vx + col;

            if nx >= SCREEN_WIDTH {
                continue;
            }

            let pixel = (sprite_row >> (7 - col as u8)) & 1;
            let i = nx + ny * SCREEN_WIDTH;

            let prev_pixel = cpu.screen_buffer[i];
            let next_pixel = prev_pixel ^ pixel;

            // If pixel changed from 1 to 0 update flag in VF.
            if prev_pixel == 1 && next_pixel == 0 {
                vf_flag = 1;
            }

            cpu.screen_buffer[i] = next_pixel;
        }
    }

    cpu.registers[0xF] = vf_flag;

    Ok(())
}

///
/// Skips instruction if the key in the Vx register is pressed.
/// If that's the case increases program counter by two.
///
fn skp_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    if cpu.keyboard[cpu.registers[x] as usize] {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Skips instruction if the key in the Vx register is not pressed.
/// If that's the case increases program counter by two.
///
fn sknp_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    if !cpu.keyboard[cpu.registers[x] as usize] {
        cpu.program_counter += 2;
    }

    Ok(())
}

///
/// Loads delay timer value into Vx register.
///
fn ld_vx_dt(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    cpu.registers[x] = cpu.delay_timer;

    Ok(())
}

///
/// Waits until any key is pressed. That key is then stored in Vx register.
///
fn ld_vx_k(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as u8;

    cpu.waiting_for_key = true;
    cpu.waiting_for_key_reg = x;

    Ok(())
}

///
/// Loads Vx to delay timer.
///
fn ld_dt_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    cpu.delay_timer = cpu.registers[x];

    Ok(())
}

///
/// Loads Vx to sound timer.
///
fn ld_st_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    cpu.sound_timer = cpu.registers[x];

    Ok(())
}

///
/// Adds Vx to index register and stores it in index register.
///
fn add_i_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    cpu.index_register += u16::from(cpu.registers[x]);

    Ok(())
}

///
/// Loads index register with address at which is stored a hexadecimal digit loaded to Vx register.
/// Returns error if there is no such digit in memory.
///
fn ld_f_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;

    if cpu.registers[x] > 0xF {
        return Err(CPUError::UnknownDigit);
    }

    cpu.index_register = u16::from(cpu.registers[x]) * 5;

    Ok(())
}

///
/// Stores BCD value of Vx register in memory starting at value in the index register.
/// Returns error if tries to write out of memory.
///
fn ld_b_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let i = cpu.index_register as usize;

    if i + 2 >= cpu.memory.len() {
        return Err(CPUError::OutOfBounds);
    }

    cpu.memory[i] = cpu.registers[x] / 100;
    cpu.memory[i + 1] = (cpu.registers[x] / 10) % 10;
    cpu.memory[i + 2] = cpu.registers[x] % 10;

    Ok(())
}

///
/// Fills memory, starting at the address in index register with values from V0 to Vx register.
/// Index register is set to "I + X + 1". Returns error if tries to write out of memory.
///
fn ld_i_vx(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let i = cpu.index_register as usize;

    if i + x >= cpu.memory.len() {
        return Err(CPUError::OutOfBounds);
    }

    cpu.memory[i..=i + x].copy_from_slice(&cpu.registers[0..=x]);

    if !cpu.load_store_quirk {
        cpu.index_register += x as u16 + 1; // Original behavior
    }

    Ok(())
}

///
/// Fills registers from V0 to Vx inclusive with memory starting at value in the index register.
/// Index register is set to "I + X + 1". Returns error if tries to read out of memory.
///
fn ld_vx_i(cpu: &mut Chip8, opcode: u16) -> CPUResult {
    let x = ((opcode & 0x0F00) >> 8) as usize;
    let i = cpu.index_register as usize;

    if i + x >= cpu.memory.len() {
        return Err(CPUError::OutOfBounds);
    }

    cpu.registers[0..=x].copy_from_slice(&cpu.memory[i..=i + x]);

    if !cpu.load_store_quirk {
        cpu.index_register += x as u16 + 1; // Original behavior
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    ///
    /// Decodes instruction, finds it in instruction table and then executes it on the cpu.
    ///
    fn run_opcode(cpu: &mut Chip8, opcode: u16) -> CPUResult {
        let id = ((opcode & 0xF000) >> 12) as usize;

        INSTRUCTION_TABLE[id](cpu, opcode)
    }

    #[test]
    fn test_cls() {
        let mut chip8 = Chip8::new();
        let buffer_size = chip8.screen_buffer.len();
        let opcode = 0x00E0;

        chip8.screen_buffer[buffer_size - 1] = 0;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert!(chip8.screen_buffer.iter().all(|v| *v == 0));
    }

    #[test]
    fn test_ret() {
        let mut chip8 = Chip8::new();
        let opcode = 0x00EE;

        // Test stack underflow
        chip8.stack_pointer = 0;
        assert!(run_opcode(&mut chip8, opcode).is_err());

        // Move to 0x124 and run CALL
        chip8.program_counter = 0x124;

        assert!(run_opcode(&mut chip8, 0x2322).is_ok());

        // Then test RET
        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.stack_pointer, 0);
        assert_eq!(chip8.program_counter, 0x124);
    }

    #[test]
    fn test_jp_addr() {
        let mut chip8 = Chip8::new();
        let opcode = 0x1123;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, 0x123);
    }

    #[test]
    fn test_call() {
        let mut chip8 = Chip8::new();

        // Test stack overflow
        let opcode = 0x2124;

        chip8.program_counter = 0;
        chip8.stack_pointer = (chip8.stack.len() - 1) as u8;

        assert!(run_opcode(&mut chip8, opcode).is_err());

        // Test proper CALL
        let opcode = 0x2322;

        chip8.program_counter = 0x124;
        chip8.stack_pointer = 0;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.stack_pointer, 1);
        assert_eq!(chip8.stack[chip8.stack_pointer as usize], 0x124);
        assert_eq!(chip8.program_counter, 0x322);
    }

    #[test]
    fn test_se_vx_byte() {
        let mut chip8 = Chip8::new();
        let opcode = 0x3203;

        // Test skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 3;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Test no skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 2;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_sne_vx_byte() {
        let mut chip8 = Chip8::new();
        let opcode = 0x4203;

        // Test skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 1;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Test no skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 3;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_se_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x5230;

        // Test skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 1;
        chip8.registers[3] = 1;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Test no skip
        let current_pc = chip8.program_counter;

        chip8.registers[3] = 2;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_ld_vx_byte() {
        let mut chip8 = Chip8::new();
        let opcode = 0x62FA;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 0xFA);
    }

    #[test]
    fn test_add_byte() {
        let mut chip8 = Chip8::new();
        let opcode = 0x720A;

        // Test no overflow
        chip8.registers[2] = 33;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 43);
        assert_eq!(chip8.registers[15], 0);

        // Test overflow
        chip8.registers[2] = 255;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 9);
        assert_eq!(chip8.registers[15], 0); // No carry generated
    }

    #[test]
    fn test_ld_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8250;

        chip8.registers[5] = 45;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 45);
    }

    #[test]
    fn test_or_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8231;

        chip8.registers[2] = 76;
        chip8.registers[3] = 123;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 127);
    }

    #[test]
    fn test_and_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8232;

        chip8.registers[2] = 76;
        chip8.registers[3] = 123;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 72);
    }

    #[test]
    fn test_xor_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8233;

        chip8.registers[2] = 76;
        chip8.registers[3] = 123;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 55);
    }

    #[test]
    fn test_add_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8234;

        // Test no overflow
        chip8.registers[2] = 120;
        chip8.registers[3] = 15;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 135);
        assert_eq!(chip8.registers[15], 0);

        // Test overflow
        chip8.registers[2] = 250;
        chip8.registers[3] = 10;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 4);
        assert_eq!(chip8.registers[15], 1);
    }

    #[test]
    fn test_sub_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8235;

        // Test no borrow
        chip8.registers[2] = 5;
        chip8.registers[3] = 2;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 3);
        assert_eq!(chip8.registers[0xF], 1);

        // Test borrow
        chip8.registers[2] = 5;
        chip8.registers[3] = 6;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 255);
        assert_eq!(chip8.registers[0xF], 0);
    }

    #[test]
    fn test_shr_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8236;

        chip8.registers[3] = 3;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 1);
        assert_eq!(chip8.registers[0xF], 1);
    }

    #[test]
    fn test_shr_vx_vy_quirked() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8236;

        chip8.registers[2] = 3;
        chip8.shift_quirk = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 1);
        assert_eq!(chip8.registers[0xF], 1);
    }

    #[test]
    fn test_subn_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x8237;

        // Test no borrow
        chip8.registers[2] = 2;
        chip8.registers[3] = 5;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 3);
        assert_eq!(chip8.registers[0xF], 1);

        // Test borrow
        chip8.registers[2] = 6;
        chip8.registers[3] = 5;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 255);
        assert_eq!(chip8.registers[0xF], 0);
    }

    #[test]
    fn test_shl_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x823E;

        chip8.registers[3] = 0xA5;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 0x4A);
        assert_eq!(chip8.registers[0xF], 1);
    }

    #[test]
    fn test_shl_vx_vy_quirked() {
        let mut chip8 = Chip8::new();
        let opcode = 0x823E;

        chip8.registers[2] = 0xA5;
        chip8.shift_quirk = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[2], 0x4A);
        assert_eq!(chip8.registers[0xF], 1);
    }

    #[test]
    fn test_ld_i_addr() {
        let mut chip8 = Chip8::new();
        let opcode = 0xA123;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.index_register, 0x123);
    }

    #[test]
    fn test_sne_vx_vy() {
        let mut chip8 = Chip8::new();
        let opcode = 0x9230;

        // Test skip
        let current_pc = chip8.program_counter;

        chip8.registers[2] = 1;
        chip8.registers[3] = 0;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Test no skip
        let current_pc = chip8.program_counter;

        chip8.registers[3] = 1;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_jp_v0_addr() {
        let mut chip8 = Chip8::new();
        let opcode = 0xBF10;

        // Test valid jump
        chip8.registers[0] = 0x0F;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, 0xF1F);

        // Test invalid jump
        chip8.registers[0] = 0xFF;

        assert!(run_opcode(&mut chip8, opcode).is_err());
        assert_eq!(chip8.program_counter, 0xF1F);
    }

    #[test]
    fn test_draw_vx_vy_nibble() {
        let mut chip8 = Chip8::new();

        chip8.index_register = 4000;
        chip8.memory[4000] = 0b01011001;
        chip8.memory[4001] = 0b10111100;

        // Test paint to screen
        let opcode = 0xD012;

        chip8.registers[0] = 12;
        chip8.registers[1] = 25;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[25 * SCREEN_WIDTH + 12..25 * SCREEN_WIDTH + 12 + 8],
            [0, 1, 0, 1, 1, 0, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[26 * SCREEN_WIDTH + 12..26 * SCREEN_WIDTH + 12 + 8],
            [1, 0, 1, 1, 1, 1, 0, 0]
        );
        assert_eq!(chip8.registers[0xF], 0);

        // Test XOR mode
        let opcode = 0xD011;

        chip8.registers[0] = 12;
        chip8.registers[1] = 25;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[25 * SCREEN_WIDTH + 12..25 * SCREEN_WIDTH + 12 + 8],
            [0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            &chip8.screen_buffer[26 * SCREEN_WIDTH + 12..26 * SCREEN_WIDTH + 12 + 8],
            [1, 0, 1, 1, 1, 1, 0, 0]
        );
        assert_eq!(chip8.registers[0xF], 1);

        // Test wrap on X-axis
        let opcode = 0xD011;

        chip8.registers[0] = 60;
        chip8.registers[1] = 1;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[1 * SCREEN_WIDTH + 60..1 * SCREEN_WIDTH + 60 + 4],
            [0, 1, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[1 * SCREEN_WIDTH + 0..1 * SCREEN_WIDTH + 0 + 4],
            [1, 0, 0, 1]
        );

        // Test wrap on Y-axis
        let opcode = 0xD012;

        chip8.registers[0] = 1;
        chip8.registers[1] = 31;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[31 * SCREEN_WIDTH + 1..31 * SCREEN_WIDTH + 1 + 8],
            [0, 1, 0, 1, 1, 0, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[0 * SCREEN_WIDTH + 1..0 * SCREEN_WIDTH + 1 + 8],
            [1, 0, 1, 1, 1, 1, 0, 0]
        );
    }

    #[test]
    fn test_draw_vx_vy_nibble_no_wrap() {
        let mut chip8 = Chip8::new();

        chip8.index_register = 4000;
        chip8.memory[4000] = 0b01011001;
        chip8.memory[4001] = 0b10111100;

        // Test paint to screen
        let opcode = 0xD012;

        chip8.registers[0] = 12;
        chip8.registers[1] = 25;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[25 * SCREEN_WIDTH + 12..25 * SCREEN_WIDTH + 12 + 8],
            [0, 1, 0, 1, 1, 0, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[26 * SCREEN_WIDTH + 12..26 * SCREEN_WIDTH + 12 + 8],
            [1, 0, 1, 1, 1, 1, 0, 0]
        );
        assert_eq!(chip8.registers[0xF], 0);

        // Test XOR mode
        let opcode = 0xD011;

        chip8.registers[0] = 12;
        chip8.registers[1] = 25;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[25 * SCREEN_WIDTH + 12..25 * SCREEN_WIDTH + 12 + 8],
            [0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            &chip8.screen_buffer[26 * SCREEN_WIDTH + 12..26 * SCREEN_WIDTH + 12 + 8],
            [1, 0, 1, 1, 1, 1, 0, 0]
        );
        assert_eq!(chip8.registers[0xF], 1);

        // Test wrap on X-axis
        let opcode = 0xD011;

        chip8.registers[0] = 60;
        chip8.registers[1] = 1;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[1 * SCREEN_WIDTH + 60..1 * SCREEN_WIDTH + 60 + 4],
            [0, 1, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[1 * SCREEN_WIDTH + 0..1 * SCREEN_WIDTH + 0 + 4],
            [0, 0, 0, 0]
        );

        // Test wrap on Y-axis
        let opcode = 0xD012;

        chip8.registers[0] = 1;
        chip8.registers[1] = 31;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(
            &chip8.screen_buffer[31 * SCREEN_WIDTH + 1..31 * SCREEN_WIDTH + 1 + 8],
            [0, 1, 0, 1, 1, 0, 0, 1]
        );
        assert_eq!(
            &chip8.screen_buffer[0 * SCREEN_WIDTH + 1..0 * SCREEN_WIDTH + 1 + 8],
            [0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_skp_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xE29E;
        chip8.registers[2] = 5;

        // Skip on pressed
        let current_pc = chip8.program_counter;

        chip8.keyboard[5] = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Don't skip on not pressed
        let current_pc = chip8.program_counter;

        chip8.keyboard[5] = false;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_sknp_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xE2A1;
        chip8.registers[2] = 5;

        // Skip on pressed
        let current_pc = chip8.program_counter;

        chip8.keyboard[5] = false;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc + 2);

        // Don't skip on not pressed
        let current_pc = chip8.program_counter;

        chip8.keyboard[5] = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.program_counter, current_pc);
    }

    #[test]
    fn test_ld_vx_dt() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF707;

        chip8.delay_timer = 0xAB;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[7], 0xAB);
    }

    #[test]
    fn test_ld_vx_k() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF10A;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert!(chip8.waiting_for_key);
        assert_eq!(chip8.waiting_for_key_reg, 1);
    }

    #[test]
    fn test_ld_dt_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF115;

        chip8.registers[1] = 0xAB;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.delay_timer, 0xAB);
    }

    #[test]
    fn test_ld_st_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF118;

        chip8.registers[1] = 0xAB;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.sound_timer, 0xAB);
    }

    #[test]
    fn test_add_i_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF11E;

        chip8.registers[1] = 0xAB;
        chip8.index_register = 0x05;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.index_register, 0xB0);
    }

    #[test]
    fn test_ld_f_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF129;

        chip8.registers[1] = 0xB;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.index_register, 0x37);

        // Error if unknown digit
        chip8.registers[1] = 0x10;

        assert!(run_opcode(&mut chip8, opcode).is_err());
        assert_eq!(chip8.index_register, 0x37);
    }

    #[test]
    fn test_ld_b_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF133;

        chip8.registers[1] = 243;
        chip8.index_register = 0x300;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.memory[0x300], 2);
        assert_eq!(chip8.memory[0x301], 4);
        assert_eq!(chip8.memory[0x302], 3);

        chip8.index_register = 0x0FFE;
        assert!(run_opcode(&mut chip8, opcode).is_err());
    }

    #[test]
    fn test_ld_i_vx() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF255;

        chip8.registers[0] = 243;
        chip8.registers[1] = 145;
        chip8.registers[2] = 5;
        chip8.index_register = 0x300;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.memory[0x300], 243);
        assert_eq!(chip8.memory[0x301], 145);
        assert_eq!(chip8.memory[0x302], 5);
        assert_eq!(chip8.index_register, 0x303);

        chip8.index_register = 0x0FFE;
        assert!(run_opcode(&mut chip8, opcode).is_err());
    }

    #[test]
    fn test_ld_i_vx_quirked() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF255;

        chip8.registers[0] = 243;
        chip8.registers[1] = 145;
        chip8.registers[2] = 5;
        chip8.index_register = 0x300;
        chip8.load_store_quirk = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.memory[0x300], 243);
        assert_eq!(chip8.memory[0x301], 145);
        assert_eq!(chip8.memory[0x302], 5);
        assert_eq!(chip8.index_register, 0x300);

        chip8.index_register = 0x0FFE;
        assert!(run_opcode(&mut chip8, opcode).is_err());
    }

    #[test]
    fn test_ld_vx_i() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF265;

        chip8.memory[0x300] = 243;
        chip8.memory[0x301] = 145;
        chip8.memory[0x302] = 5;
        chip8.index_register = 0x300;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[0], 243);
        assert_eq!(chip8.registers[1], 145);
        assert_eq!(chip8.registers[2], 5);
        assert_eq!(chip8.index_register, 0x303);

        chip8.index_register = 0x0FFE;
        assert!(run_opcode(&mut chip8, opcode).is_err());
    }

    #[test]
    fn test_ld_vx_i_quirked() {
        let mut chip8 = Chip8::new();
        let opcode = 0xF265;

        chip8.memory[0x300] = 243;
        chip8.memory[0x301] = 145;
        chip8.memory[0x302] = 5;
        chip8.index_register = 0x300;
        chip8.load_store_quirk = true;

        assert!(run_opcode(&mut chip8, opcode).is_ok());
        assert_eq!(chip8.registers[0], 243);
        assert_eq!(chip8.registers[1], 145);
        assert_eq!(chip8.registers[2], 5);
        assert_eq!(chip8.index_register, 0x300);

        chip8.index_register = 0x0FFE;
        assert!(run_opcode(&mut chip8, opcode).is_err());
    }
}
