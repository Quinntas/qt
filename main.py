import argparse
import re
import subprocess
from dataclasses import dataclass
from enum import Enum, auto

from llvmlite.binding import *


class TokenType(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    BOOL = auto()
    VECTOR = auto()
    MAP = auto()
    PRINT = auto()

    ID = auto()
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    CHAR_LIT = auto()

    EQUALS = auto()  # =
    PLUS = auto()  # +
    COMMA = auto()  # ,
    SEMI = auto()  # ;
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACE = auto()  # {
    RBRACE = auto()  # }


@dataclass
class Token:
    type: TokenType
    value: any = None


class Lexer:
    keywords = {
        'int': TokenType.INT,
        'float': TokenType.FLOAT,
        'string': TokenType.STRING,
        'char': TokenType.CHAR,
        'bool': TokenType.BOOL,
        'vector': TokenType.VECTOR,
        'map': TokenType.MAP,
        'print': TokenType.PRINT
    }

    punctuation_map = {
        '=': TokenType.EQUALS,
        '+': TokenType.PLUS,
        ',': TokenType.COMMA,
        ';': TokenType.SEMI,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE
    }

    def __init__(self):
        token_spec = [
            ('NUMBER', r'\d+(\.\d*)?'),
            ('STRING', r'"([^"\\]|\\.)*"'),
            ('CHAR', r"'.'"),
            ('ID', r'[a-zA-Z_]\w*'),
            ('PUNCT', r'[=\+,\;\[\]\(\)\{\}]'),
            ('COMMENTS', r'//.*'),
            ('WHITESPACE', r'\s+')
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)
        self.token_re = re.compile(tok_regex)

    def tokenize(self, code):
        for mo in self.token_re.finditer(code):
            kind = mo.lastgroup
            value = mo.group()
            if kind in ('WHITESPACE', 'COMMENTS'):
                continue
            elif kind == 'ID':
                token_type = self.keywords.get(value, TokenType.ID)
                yield Token(token_type, value)
            elif kind == 'NUMBER':
                if '.' in value:
                    yield Token(TokenType.FLOAT_LIT, float(value))
                else:
                    yield Token(TokenType.INT_LIT, int(value))
            elif kind == 'STRING':
                yield Token(TokenType.STRING_LIT, value[1:-1])
            elif kind == 'CHAR':
                yield Token(TokenType.CHAR_LIT, value[1])
            elif kind == 'PUNCT':
                token_type = self.punctuation_map.get(value)
                if token_type is None:
                    raise ValueError("Unknown punctuation: " + value)
                yield Token(token_type, value)
            else:
                raise ValueError("Unknown token kind: " + kind)


class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def match(self, token_type):
        if self.current_token and self.current_token.type == token_type:
            current = self.current_token
            self.advance()
            return current
        else:
            raise SyntaxError(f"Expected token {token_type} but got {self.current_token}")

    def parse(self):
        return self.program()

    def program(self):
        stmts = []
        while self.current_token:
            stmts.append(self.stmt())
        return stmts

    def stmt(self):
        if self.current_token.type == TokenType.PRINT:
            return self.print_stmt()
        else:
            return self.decl()

    def decl(self):
        type_info = self.type_spec()
        ident = self.match(TokenType.ID).value
        self.match(TokenType.EQUALS)
        expr = self.expr()
        self.match(TokenType.SEMI)
        return ('decl', type_info, ident, expr)

    def type_spec(self):
        base_type = self.current_token.type
        self.advance()
        return base_type

    def print_stmt(self):
        self.match(TokenType.PRINT)
        self.match(TokenType.LPAREN)
        expr = self.expr()
        self.match(TokenType.RPAREN)
        self.match(TokenType.SEMI)
        return ('print', expr)

    def expr(self):
        node = self.term()
        while self.current_token and self.current_token.type == TokenType.PLUS:
            self.match(TokenType.PLUS)
            right = self.term()
            node = ('+', node, right)
        return node

    def term(self):
        token = self.current_token
        if token.type in (TokenType.INT_LIT, TokenType.FLOAT_LIT,
                          TokenType.STRING_LIT, TokenType.CHAR_LIT):
            self.advance()
            return token.value
        elif token.type == TokenType.ID:
            self.advance()
            return token.value
        elif token.type == TokenType.LPAREN:
            self.match(TokenType.LPAREN)
            node = self.expr()
            self.match(TokenType.RPAREN)
            return node
        else:
            raise SyntaxError("Unexpected token in term: " + str(token))


class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}

    def analyze(self, ast):
        for node in ast:
            if node[0] == 'decl':
                _, type_info, ident, expr = node
                self.check_type(type_info, expr)
                self.symbol_table[ident] = type_info

    def check_type(self, decl_type, expr):
        expr_type = self.get_expr_type(expr)
        if decl_type != expr_type:
            raise TypeError(f"Type mismatch: {decl_type} vs {expr_type}")

    def get_expr_type(self, expr):
        if isinstance(expr, int):
            return TokenType.INT
        elif isinstance(expr, float):
            return TokenType.FLOAT
        elif isinstance(expr, tuple):
            return self.get_expr_type(expr[1])
        else:
            return None


class Compiler:
    def __init__(self, entry_point="entry_point"):
        self.module = ir.Module(name="module")
        self.builder = None
        self.symbol_table = {}
        self.entry_point = entry_point
        self.printf_func = None
        self.format_string_cache = {}

    def compile(self, ast):
        func_type = ir.FunctionType(ir.VoidType(), [])
        entry_fn = ir.Function(self.module, func_type, name=self.entry_point)
        block = entry_fn.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        for node in ast:
            self.compile_node(node)

        self.builder.ret_void()

        if self.entry_point != "main":
            wrapper_type = ir.FunctionType(ir.IntType(32), [])
            main_fn = ir.Function(self.module, wrapper_type, name="main")
            entry_block = main_fn.append_basic_block("entry")
            wrapper_builder = ir.IRBuilder(entry_block)
            wrapper_builder.call(entry_fn, [])
            wrapper_builder.ret(ir.Constant(ir.IntType(32), 0))

        return self.module

    def compile_node(self, node):
        if node[0] == 'print':
            self.compile_print(node[1])
        elif node[0] == 'decl':
            self.compile_decl(node)

    def compile_print(self, expr):
        value = self.compile_expr(expr)
        printf = self.get_printf()
        if value.type == ir.IntType(32):
            fmt_ptr = self.get_or_create_global_string("%d\n", "fmt_int")
        elif value.type == ir.DoubleType():
            fmt_ptr = self.get_or_create_global_string("%f\n", "fmt_float")
        else:
            raise NotImplementedError("Print not implemented for type: " + str(value.type))
        self.builder.call(printf, [fmt_ptr, value])

    def compile_decl(self, node):
        _, type_info, ident, expr = node
        value = self.compile_expr(expr)
        alloca = self.builder.alloca(self.llvm_type(type_info), name=ident)
        self.builder.store(value, alloca)
        self.symbol_table[ident] = alloca

    def llvm_type(self, type_token):
        mapping = {
            TokenType.INT: ir.IntType(32),
            TokenType.FLOAT: ir.DoubleType(),
        }
        return mapping.get(type_token, ir.IntType(32))

    def compile_expr(self, expr):
        if isinstance(expr, int):
            return ir.Constant(ir.IntType(32), expr)
        elif isinstance(expr, float):
            return ir.Constant(ir.DoubleType(), expr)
        elif isinstance(expr, str):
            alloca = self.symbol_table.get(expr)
            if alloca is None:
                raise NameError("Undefined variable: " + expr)
            return self.builder.load(alloca, name=expr + "_load")
        elif isinstance(expr, tuple):
            op = expr[0]
            left = self.compile_expr(expr[1])
            right = self.compile_expr(expr[2])
            if op == '+':
                if left.type == ir.IntType(32):
                    return self.builder.add(left, right, name="addtmp")
                elif left.type == ir.DoubleType():
                    return self.builder.fadd(left, right, name="addtmp")
            raise NotImplementedError("Operator not implemented: " + op)
        else:
            raise NotImplementedError("Expression compilation not implemented for: " + str(expr))

    def get_printf(self):
        if self.printf_func:
            return self.printf_func
        printf_func = self.module.globals.get("printf")
        if printf_func is None:
            voidptr_ty = ir.IntType(8).as_pointer()
            printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
            printf_func = ir.Function(self.module, printf_ty, name="printf")
        self.printf_func = printf_func
        return printf_func

    def get_or_create_global_string(self, string, name):
        if name in self.format_string_cache:
            return self.format_string_cache[name]
        if name in self.module.globals:
            global_var = self.module.globals[name]
        else:
            str_val = bytearray(string.encode("utf8"))
            str_type = ir.ArrayType(ir.IntType(8), len(str_val) + 1)
            global_var = ir.GlobalVariable(self.module, str_type, name=name)
            global_var.linkage = 'internal'
            global_var.global_constant = True
            global_var.initializer = ir.Constant(str_type, list(str_val) + [0])
        fmt_ptr = self.builder.bitcast(global_var, ir.IntType(8).as_pointer())
        self.format_string_cache[name] = fmt_ptr
        return fmt_ptr


def main():
    arg_parser = argparse.ArgumentParser(
        description="Compile a simple language to LLVM IR and an executable."
    )
    arg_parser.add_argument('--entry', '-e', type=str, default="entry_point",
                            help="Path to the source file (or a string if no file exists) for the program entry point.")
    arg_parser.add_argument('--out_dir', '-o', type=str, default=".",
                            help="Output directory for IR and binary (default: current directory)")
    args = arg_parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.isfile(args.entry):
        with open(args.entry, "r") as f:
            code = f.read()
        entry_func_name = "entry_point"
    else:
        code = args.entry
        entry_func_name = args.entry

    # Lexical Analysis.
    lexer = Lexer()
    tokens = list(lexer.tokenize(code))

    # Syntax Analysis.
    parser = Parser(tokens)
    ast = parser.parse()

    # Semantic Analysis.
    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)

    # IR Generation.
    compiler = Compiler(entry_point=entry_func_name)
    module = compiler.compile(ast)

    print("Generated LLVM IR:")
    print(module)

    # Write the IR to a file.
    ir_file = os.path.join(args.out_dir, "output.ll")
    with open(ir_file, "w") as f:
        f.write(str(module))

    # Compile with clang using optimization flag.
    out_binary = os.path.join(args.out_dir, "output")
    clang_cmd = ["clang", "-O3", "-o", out_binary, ir_file]
    subprocess.run(clang_cmd, check=True)
    print(f"Compilation complete. Binary generated at: {out_binary}")


if __name__ == "__main__":
    main()
