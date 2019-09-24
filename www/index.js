import { Chip8Emulator, keyMap } from "./chip8";

const canvas = document.getElementById('canvas');
const rom_selector = document.getElementById("select");
const description = document.getElementById("game-description");

// Canvas
const ctx = canvas.getContext('2d');
const ALIVE_COLOR = "#FFFFFF";
const DEAD_COLOR = "#000000";
const x_scaler = Math.floor(canvas.width / 64);
const y_scaler = Math.floor(canvas.height / 32);


// Emulator
let emulator = new Chip8Emulator();
let t1 = 0;
let t2 = 0;
let roms = null;
let requestID = null;

fetch("/roms.json")
    .then(response => {
        if(!response.ok) {
            throw new Error('Could not get roms.json file.');
        }
        return response.json();
    })
    .then(json => {
        roms = json['roms'];

        for (let i = 0; i < roms.length; i++) {
            let opt = document.createElement('option');
            opt.appendChild(document.createTextNode(roms[i]['name']));
            opt.value = i.toString();
            rom_selector.appendChild(opt)
        }
    });

const clearScreen = () => {
    ctx.beginPath();

    ctx.fillStyle = DEAD_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.stroke();
};

const emulation = () => {
    // Calculate how many cycles of the CPU we need to perform
    // in this "emulation" call to achieve given frequency.
    t2 = performance.now();
    let delta = t2 - t1;

    emulator.step(delta);

    // Draw screen buffer to the canvas.
    drawScreen();

    // Save time of the last call
    t1 = t2;

    requestID = requestAnimationFrame(emulation);
};

const getIndex = (row, column) => {
    return row * 64 + column;
};

const drawScreen = () => {
    const pixels = emulator.pixelBuffer;

    ctx.beginPath();

    ctx.fillStyle = DEAD_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = ALIVE_COLOR;
    for (let i = 0; i < 32; i++) {
        for (let j = 0; j < 64; j++) {
            let idx = getIndex(i, j);
            if (pixels[idx] !== 1) {
                continue;
            }
            ctx.fillRect(j * x_scaler, i * y_scaler, x_scaler, y_scaler);
        }
    }

    ctx.stroke();
};

document.addEventListener('keydown', (event) => {
    const keyName = event.key;
    const keyID = keyMap.get(keyName);

    if (keyID !== undefined)
        emulator.setKey(keyID, true);
});

document.addEventListener('keyup', (event) => {
    const keyName = event.key;
    const keyID = keyMap.get(keyName);

    if (keyID !== undefined)
        emulator.setKey(keyID, false);
});

rom_selector.addEventListener('change', (event) => {
    rom_selector.blur();
    let rom_data = roms[event.target.value];

    if (rom_data === undefined) {
        cancelAnimationFrame(requestID);
        requestAnimationFrame(clearScreen);
        emulator.reset();
        description.textContent = '';
    } else {
        fetch(rom_data.location)
            .then(response => {
                if(!response.ok)
                    throw new Error('Could not get ' + rom_data['name'] + ' rom file.');
                return response.arrayBuffer();
            })
            .then(buffer => {
                description.textContent = rom_data['description'];
                let rom = new Uint8Array(buffer);
                emulator.load(rom_data, rom);
                t1 = performance.now();
                requestID = requestAnimationFrame(emulation);
            });
    }
});
