# Intro

This repo is a fork on PINNacle to research in PINN under limited data

## Get Started

#### Step 1: Install Dependencies Manager Pixi

Linux/MaxOS

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows

```
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

#### Step 2: Run pixi install

```sh
pixi install
```

### Step 3: Run Benchmark

`pixi shell` command uses the pixi env

`python benchmark_row.py --device cpu` runs the benchmark row with cpu device
