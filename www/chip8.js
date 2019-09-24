import { Chip8 } from "wasm-chip8-emulator";
import { memory } from "wasm-chip8-emulator/wasm_chip8_emulator_bg";

// Mappings from normal keyboard to Chip8 keyboard
export const keyMap = new Map([
    ['1', 1],
    ['2', 2],
    ['3', 3],
    ['4', 12],
    ['q', 4],
    ['w', 5],
    ['e', 6],
    ['r', 13],
    ['a', 7],
    ['s', 8],
    ['d', 9],
    ['f', 14],
    ['z', 10],
    ['x', 0],
    ['c', 11],
    ['v', 15]
]);

export class Chip8Emulator {
    constructor() {
        this.cpu = Chip8.new();
        this.frequency = 0;
        this.cpu_cycles = 0;
        this.timer_cycles = 0;

        this.audioCtx = new AudioContext();
        this.oscillator = this.audioCtx.createOscillator();
        this.oscillator.type = 'square';
        this.oscillator.frequency.value = 200;
        this.oscillator.start(0);
    }

    step(delta) {
        this.cycleCPU(delta);
        this.updateTimers(delta);

        if(this.cpu.sound()) {
            this.playSound();
        } else {
            this.stopSound();
        }
    }

    cycleCPU(delta) {
        this.cpu_cycles += (delta * this.frequency) / 1000;

        // Execute integer amount of the cycles. Fraction cycles will be passed onto the next call.
        let current_cycles = Math.floor(this.cpu_cycles);
        for (let i = 0; i < current_cycles; i++) {
            this.cpu.cycle();
        }
        this.cpu_cycles -= current_cycles;
    }

    updateTimers(delta) {
        this.timer_cycles += (delta * 60) / 1000;
        let current_cycles = Math.floor(this.timer_cycles);
        for (let i = 0; i < current_cycles; i++) {
            this.cpu.update_timers();
        }
        this.timer_cycles -= current_cycles;
    }

    load(rom_config, rom) {
        this.reset();

        this.frequency = rom_config.speed;
        if(rom_config['load_store_quirk'] !== undefined)
            this.cpu.set_load_store_quirk(rom_config['load_store_quirk']);
        if(rom_config['shift_quirk'] !== undefined)
            this.cpu.set_shift_quirk(rom_config['shift_quirk']);
        if(rom_config['wrapping'] !== undefined)
            this.cpu.set_wrapping(rom_config['wrapping']);
        this.cpu.load(rom);
    }

    reset() {
        this.cpu_cycles = 0;
        this.frequency = 0;
        this.oscillator.disconnect();
        this.cpu.reset();
    }

    playSound() {
        this.oscillator.connect(this.audioCtx.destination);
    }

    stopSound() {
        this.oscillator.disconnect();
    }

    setKey(id, value) {
        this.cpu.set_key(id, value);
    }

    get pixelBuffer() {
        const screenBufferPtr = this.cpu.screen_buffer();
        return new Uint8Array(memory.buffer, screenBufferPtr, 64 * 32);
    }
}