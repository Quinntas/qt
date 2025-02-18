# Simple LLVM Compiler

This is a lightweight compiler written in Python that compiles a simple custom language to LLVM IR and produces an
executable using Clang. The project uses the [llvmlite](https://llvmlite.readthedocs.io/) library for LLVM IR generation
and supports basic operations such as integer arithmetic, variable declarations, and print statements.

## Features

- **Lexical Analysis:** Uses precompiled regex patterns and static dictionaries for fast, O(1) token lookups.
- **Syntax Analysis:** Implements a recursive descent parser for a simple language.
- **Semantic Analysis:** Performs basic type checking.
- **IR Generation:** Generates LLVM IR for the parsed AST and uses a wrapper `main` function to satisfy linker
  requirements.
- **Optimizations:** Caches IR constructs (like `printf` and format strings) to speed up compilation.
- **Command-Line Interface:** Accepts command-line arguments to specify the source file or code string, entry function
  name, and output directory.
- **Executable Generation:** Compiles the generated LLVM IR with Clang using optimization flags (e.g., `-O3`).

## Requirements

- Python 3.6 or higher
- [llvmlite](https://llvmlite.readthedocs.io/) (Install via `pip install llvmlite`)
- Clang (for linking and creating the final executable)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/simple-llvm-compiler.git
   cd simple-llvm-compiler
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, you can install `llvmlite` directly:*

   ```bash
   pip install llvmlite
   ```

## Usage

Run the compiler using the `main.py` script. You can specify the source code file (or source string) using the `-e`/
`--entry` argument and the output directory using the `-o`/`--out_dir` argument.

### Example

Compile a source file located at `./tests/math.qt` and output the generated LLVM IR and binary to the `./out` directory:

```bash
python main.py -e ./tests/math.qt -o ./out
```

### Command-Line Arguments

- `-e` / `--entry`:  
  Specifies the source file path or a string containing the source code. If a valid file path is provided, the file's
  contents will be used as the source code. Otherwise, the string itself will be treated as the source code.

- `-o` / `--out_dir`:  
  Specifies the output directory where the LLVM IR (`output.ll`) and the final executable (`output`) will be generated.

## Project Structure

```
simple-llvm-compiler/
├── main.py         # Main driver code for the compiler
├── README.md       # This file
├── tests/          # Directory for test source files (e.g., math.qt)
└── out/            # Output directory for generated IR and executable (created at runtime)
```

## How It Works

1. **Lexical Analysis:**  
   The lexer tokenizes the input source code using a precompiled regular expression. It uses static dictionaries for
   keywords and punctuation to ensure fast token lookup.

2. **Syntax Analysis:**  
   The parser performs recursive descent parsing of the token stream, creating an abstract syntax tree (AST) for the
   program.

3. **Semantic Analysis:**  
   Basic type checking is performed by traversing the AST. The symbol table is updated with variable declarations.

4. **IR Generation:**  
   The compiler generates LLVM IR from the AST. A fixed entry function (e.g., `entry_point`) is generated to hold the
   user code, and a wrapper `main` function is created to call this entry point.

5. **Executable Generation:**  
   The generated LLVM IR is written to `output.ll`, and Clang is invoked with optimization flags (e.g., `-O3`) to
   produce an optimized executable.

## Troubleshooting

- **Linker Error (`undefined reference to 'main'`):**  
  Ensure that the wrapper `main` function is generated. If you are providing a custom entry point, the compiler creates
  a `main` function that calls your entry point.

- **Clang Not Found:**  
  Make sure Clang is installed and available in your system's PATH.

## Contributing

Contributions, bug fixes, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

Happy compiling!