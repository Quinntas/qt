#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any

from llvmlite import ir


class TokenType(Enum):
    __slots__ = ()

    # Keywords / Types
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    BOOL = auto()
    VECTOR = auto()
    MAP = auto()
    PRINT = auto()
    RETURN = auto()
    IMPORT = auto()
    FROM = auto()
    EXPORT = auto()

    # Identifiers and Literals
    ID = auto()
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    CHAR_LIT = auto()

    # Operators and punctuation
    EQUALS = auto()
    PLUS = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()
    COMMA = auto()
    SEMI = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()


@dataclass()
class Token:
    type: TokenType
    value: Any = None


class Lexer:
    __slots__ = ['token_re', 'token_patterns']

    keywords: Dict[str, TokenType] = {
        'int': TokenType.INT,
        'float': TokenType.FLOAT,
        'string': TokenType.STRING,
        'char': TokenType.CHAR,
        'bool': TokenType.BOOL,
        'vector': TokenType.VECTOR,
        'map': TokenType.MAP,
        'print': TokenType.PRINT,
        'return': TokenType.RETURN,
        'import': TokenType.IMPORT,
        'from': TokenType.FROM,
        'export': TokenType.EXPORT,
    }

    punctuation_map: Dict[str, TokenType] = {
        '=': TokenType.EQUALS,
        '+': TokenType.PLUS,
        '-': TokenType.SUB,
        '*': TokenType.MULT,
        '/': TokenType.DIV,
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
        # Optimize regex patterns
        token_spec = [
            ('NUMBER', r'\d+(?:\.\d*)?'),
            ('STRING', r'"[^"\\]*(?:\\.[^"\\]*)*"'),
            ('CHAR', r"'.'"),
            ('ID', r'[a-zA-Z_]\w*'),
            ('PUNCT', r'[=+\-*/,;[\](){}]'),
            ('COMMENTS', r'//[^\n]*'),
            ('WHITESPACE', r'\s+')
        ]
        self.token_re = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec))
        self.token_patterns = dict(token_spec)

    def tokenize(self, code: str) -> List[Token]:
        tokens = []
        for mo in self.token_re.finditer(code):
            kind = mo.lastgroup
            value = mo.group()

            if kind in ('WHITESPACE', 'COMMENTS'):
                continue

            if kind == 'ID':
                token_type = self.keywords.get(value, TokenType.ID)
                tokens.append(Token(token_type, value))
            elif kind == 'NUMBER':
                if '.' in value:
                    tokens.append(Token(TokenType.FLOAT_LIT, float(value)))
                else:
                    tokens.append(Token(TokenType.INT_LIT, int(value)))
            elif kind == 'STRING':
                tokens.append(Token(TokenType.STRING_LIT, value[1:-1]))
            elif kind == 'CHAR':
                tokens.append(Token(TokenType.CHAR_LIT, value[1]))
            elif kind == 'PUNCT':
                token_type = self.punctuation_map.get(value)
                if token_type is None:
                    raise ValueError(f"Unknown punctuation: {value}")
                tokens.append(Token(token_type, value))
            else:
                raise ValueError(f"Unknown token kind: {kind}")

        return tokens


class Parser:
    __slots__ = ['tokens', 'pos', 'current_token']

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None

    def peek(self, offset: int = 1) -> Optional[Token]:
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else None

    def advance(self) -> None:
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def error(self, message: str):
        raise SyntaxError(f"Syntax error at position {self.pos}: {message}")

    def match(self, token_type: TokenType) -> Token:
        if self.current_token and self.current_token.type == token_type:
            current = self.current_token
            self.advance()
            return current
        self.error(f"Expected {token_type.name}, got {self.current_token.type.name if self.current_token else 'EOF'}")

    def program(self) -> List[Tuple]:
        nodes = []
        while self.current_token:
            if self.current_token.type == TokenType.IMPORT:
                nodes.append(self.import_stmt())
            elif self.current_token.type == TokenType.EXPORT:
                nodes.append(self.export_decl())
            elif self.current_token.type in {TokenType.INT, TokenType.FLOAT, TokenType.STRING,
                                             TokenType.CHAR, TokenType.BOOL}:
                peek_token = self.peek()
                if (peek_token and peek_token.type == TokenType.ID and
                        self.peek(2) and self.peek(2).type == TokenType.LPAREN):
                    nodes.append(self.function_def())
                else:
                    nodes.append(self.decl())
            else:
                nodes.append(self.stmt())
        return nodes

    def import_stmt(self) -> Tuple:
        self.match(TokenType.IMPORT)
        self.match(TokenType.LBRACE)
        names = [self.match(TokenType.ID).value]
        while self.current_token and self.current_token.type == TokenType.COMMA:
            self.match(TokenType.COMMA)
            names.append(self.match(TokenType.ID).value)
        self.match(TokenType.RBRACE)
        self.match(TokenType.FROM)
        filename = self.match(TokenType.STRING_LIT).value
        self.match(TokenType.SEMI)
        return ('import', names, filename)

    def export_decl(self) -> Tuple:
        self.match(TokenType.EXPORT)
        type_info = self.type_spec()
        ident = self.match(TokenType.ID).value
        self.match(TokenType.EQUALS)
        expr = self.expr()
        self.match(TokenType.SEMI)
        return ('export_decl', type_info, ident, expr)

    def type_spec(self) -> TokenType:
        t = self.current_token.type
        self.advance()
        return t

    def stmt(self) -> Tuple:
        if self.current_token.type == TokenType.PRINT:
            return self.print_stmt()
        elif self.current_token.type == TokenType.RETURN:
            return self.return_stmt()
        elif self.current_token.type == TokenType.LBRACE:
            return self.compound_stmt()
        else:
            expr = self.expr()
            self.match(TokenType.SEMI)
            return expr

    def compound_stmt(self) -> Tuple:
        self.match(TokenType.LBRACE)
        stmts = []
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            stmts.append(self.stmt())
        self.match(TokenType.RBRACE)
        return ('compound', stmts)

    def function_def(self) -> Tuple:
        try:
            ret_type = self.type_spec()
            func_name = self.match(TokenType.ID).value
            self.match(TokenType.LPAREN)
            params = []
            if self.current_token.type != TokenType.RPAREN:
                params = self.param_list()
            self.match(TokenType.RPAREN)
            body = self.compound_stmt()
            return ('func_def', ret_type, func_name, params, body)
        except SyntaxError as e:
            print(f"Error in function definition: {e}")
            raise

    def param_list(self) -> List[Tuple]:
        params = [self.param()]
        while self.current_token and self.current_token.type == TokenType.COMMA:
            self.match(TokenType.COMMA)
            params.append(self.param())
        return params

    def param(self) -> Tuple:
        param_type = self.type_spec()
        param_name = self.match(TokenType.ID).value
        return (param_type, param_name)

    def print_stmt(self) -> Tuple:
        self.match(TokenType.PRINT)
        self.match(TokenType.LPAREN)
        expr = self.expr()
        self.match(TokenType.RPAREN)
        self.match(TokenType.SEMI)
        return ('print', expr)

    def return_stmt(self) -> Tuple:
        self.match(TokenType.RETURN)
        expr = self.expr()
        self.match(TokenType.SEMI)
        return ('return', expr)

    def decl(self) -> Tuple:
        type_info = self.type_spec()
        ident = self.match(TokenType.ID).value
        self.match(TokenType.EQUALS)
        expr = self.expr()
        self.match(TokenType.SEMI)
        return ('decl', type_info, ident, expr)

    def expr(self) -> Any:
        return self.add_expr()

    def add_expr(self) -> Any:
        node = self.mult_expr()
        while self.current_token and self.current_token.type in {TokenType.PLUS, TokenType.SUB}:
            if self.current_token.type == TokenType.PLUS:
                self.match(TokenType.PLUS)
                right = self.mult_expr()
                node = ('+', node, right)
            else:
                self.match(TokenType.SUB)
                right = self.mult_expr()
                node = ('-', node, right)
        return node

    def mult_expr(self) -> Any:
        node = self.factor()
        while self.current_token and self.current_token.type in {TokenType.MULT, TokenType.DIV}:
            if self.current_token.type == TokenType.MULT:
                self.match(TokenType.MULT)
                right = self.factor()
                node = ('*', node, right)
            elif self.current_token.type == TokenType.DIV:
                self.match(TokenType.DIV)
                right = self.factor()
                node = ('/', node, right)
        return node

    def factor(self) -> Any:
        token = self.current_token
        if token.type in (TokenType.INT_LIT, TokenType.FLOAT_LIT, TokenType.STRING_LIT, TokenType.CHAR_LIT):
            self.advance()
            return token.value
        elif token.type == TokenType.ID:
            self.advance()
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                self.match(TokenType.LPAREN)
                args = []
                if self.current_token.type != TokenType.RPAREN:
                    args.append(self.expr())
                    while self.current_token and self.current_token.type == TokenType.COMMA:
                        self.match(TokenType.COMMA)
                        args.append(self.expr())
                self.match(TokenType.RPAREN)
                return ('call', token.value, args)
            else:
                return token.value
        elif token.type == TokenType.LPAREN:
            self.match(TokenType.LPAREN)
            node = self.expr()
            self.match(TokenType.RPAREN)
            return node
        else:
            raise SyntaxError(f"Unexpected token in factor: {token}")


class SemanticAnalyzer:
    __slots__ = ['symbol_table']

    def __init__(self):
        self.symbol_table: Dict[str, Any] = {}

    def analyze(self, ast: List[Tuple]) -> None:
        for node in ast:
            if node[0] in ('decl', 'export_decl'):
                _, type_info, ident, expr = node
                self.symbol_table[ident] = type_info
            elif node[0] == 'func_def':
                _, ret_type, func_name, params, _ = node
                self.symbol_table[func_name] = (ret_type, params)


class Compiler:
    __slots__ = ['module', 'builder', 'scopes', 'printf_func', 'format_string_cache']

    def __init__(self, module: Optional[ir.Module] = None):
        self.module = module if module is not None else ir.Module(name="module")
        self.builder: Optional[ir.IRBuilder] = None
        self.scopes: List[Dict[str, Any]] = [{}]
        self.printf_func: Optional[ir.Function] = None
        self.format_string_cache: Dict[str, ir.Constant] = {}

    def compile_globals(self, ast: List[Tuple], init_fn_name: str) -> ir.Function:
        init_fn = ir.Function(self.module,
                              ir.FunctionType(ir.VoidType(), []),
                              name=init_fn_name)

        block = init_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)

        self.scopes.append({})

        for node in ast:
            if isinstance(node, tuple) and node[0] == 'import':
                continue
            self.compile_node(node)

        self.builder.ret_void()

        self.scopes.pop()

        return init_fn

    def compile_node(self, node: Any) -> Optional[ir.Value]:
        if isinstance(node, tuple):
            tag = node[0]
            if tag == 'print':
                return self.compile_print(node[1])
            elif tag in ('decl', 'export_decl'):
                return self.compile_decl(node)
            elif tag == 'func_def':
                return self.compile_func_def(node)
            elif tag == 'return':
                return self.compile_return(node)
            elif tag == 'compound':
                self.scopes.append({})
                for stmt in node[1]:
                    self.compile_node(stmt)
                self.scopes.pop()
                return None
            elif tag in ('+', '-', '*', '/', 'call'):
                return self.compile_expr(node)
            else:
                return self.compile_expr(node)
        else:
            return self.compile_expr(node)

    def compile_decl(self, node: Tuple) -> Optional[ir.Value]:
        tag, type_info, ident, expr = node
        value = self.compile_expr(expr)
        llvm_ty = self.llvm_type(type_info)

        if isinstance(value, ir.Constant):
            global_var = ir.GlobalVariable(self.module, llvm_ty, ident)
            if tag == 'export_decl':
                # Default linkage for exports (global definition)
                pass
            else:
                global_var.linkage = 'internal'
            global_var.global_constant = True
            global_var.initializer = value
            self.scopes[-1][ident] = global_var
            if tag == 'export_decl':
                self.scopes[0][ident] = global_var
        else:
            alloca = self.builder.alloca(llvm_ty, name=ident)
            self.builder.store(value, alloca)
            self.scopes[-1][ident] = alloca

        return None

    def compile_func_def(self, node: Tuple) -> Optional[ir.Function]:
        _, ret_type, func_name, params, body = node
        llvm_ret_type = self.llvm_type(ret_type)
        llvm_param_types = [self.llvm_type(p[0]) for p in params]

        # Create function type and function
        func_type = ir.FunctionType(llvm_ret_type, llvm_param_types)
        llvm_function = ir.Function(self.module, func_type, name=func_name)

        # Add to global scope
        self.scopes[0][func_name] = llvm_function

        # Create entry block
        entry_block = llvm_function.append_basic_block("entry")
        old_builder = self.builder
        self.builder = ir.IRBuilder(entry_block)

        # Create new scope for function
        self.scopes.append({})

        # Handle parameters
        for i, arg in enumerate(llvm_function.args):
            param_name = params[i][1]
            arg.name = param_name
            alloc = self.builder.alloca(arg.type, name=param_name)
            self.builder.store(arg, alloc)
            self.scopes[-1][param_name] = alloc

        # Compile function body
        self.compile_node(body)

        # Add default return if needed
        if not self.builder.block.is_terminated:
            if isinstance(llvm_ret_type, ir.VoidType):
                self.builder.ret_void()
            elif llvm_ret_type == ir.IntType(32):
                self.builder.ret(ir.Constant(ir.IntType(32), 0))
            elif llvm_ret_type == ir.DoubleType():
                self.builder.ret(ir.Constant(ir.DoubleType(), 0.0))
            else:
                self.builder.ret_void()

        # Restore previous state
        self.scopes.pop()
        self.builder = old_builder

        return llvm_function

    def compile_print(self, expr: Any) -> Optional[ir.Value]:
        value = self.compile_expr(expr)
        printf = self.get_printf()

        # Add type checking for print argument
        if not isinstance(value.type, (ir.IntType, ir.DoubleType)):
            raise TypeError(f"Cannot print value of type {value.type}")

        # Handle different types
        if isinstance(value.type, ir.IntType):
            fmt_ptr = self.get_or_create_global_string("%d\n", "fmt_int")
        elif isinstance(value.type, ir.DoubleType):
            fmt_ptr = self.get_or_create_global_string("%f\n", "fmt_float")
        else:
            raise TypeError(f"Unsupported print type: {value.type}")

        return self.builder.call(printf, [fmt_ptr, value])

    def compile_return(self, node: Tuple) -> Optional[ir.Value]:
        ret_val = self.compile_expr(node[1])
        return self.builder.ret(ret_val)

    def compile_expr(self, expr: Any) -> ir.Value:
        if isinstance(expr, (int, float)):
            if isinstance(expr, int):
                return ir.Constant(ir.IntType(32), expr)
            else:
                return ir.Constant(ir.DoubleType(), expr)
        elif isinstance(expr, str):
            # Look up variable in all scopes
            for scope in reversed(self.scopes):
                if expr in scope:
                    var = scope[expr]
                    return self.builder.load(var, name=f"{expr}_load")
            # Fallback: check module-level globals
            if expr in self.module.globals:
                var = self.module.globals[expr]
                return self.builder.load(var, name=f"{expr}_load")
            raise NameError(f"Undefined variable: {expr}")
        elif isinstance(expr, tuple):
            tag = expr[0]
            if tag == '+':
                left = self.compile_expr(expr[1])
                right = self.compile_expr(expr[2])
                if left.type == ir.IntType(32):
                    return self.builder.add(left, right, name="addtmp")
                elif left.type == ir.DoubleType():
                    return self.builder.fadd(left, right, name="addtmp")
                else:
                    raise NotImplementedError(f"Addition not implemented for type: {left.type}")
            elif tag == '-':
                left = self.compile_expr(expr[1])
                right = self.compile_expr(expr[2])
                if left.type == ir.IntType(32):
                    return self.builder.sub(left, right, name="subtmp")
                elif left.type == ir.DoubleType():
                    return self.builder.fsub(left, right, name="subtmp")
                else:
                    raise NotImplementedError(f"Subtraction not implemented for type: {left.type}")
            elif tag == '*':
                left = self.compile_expr(expr[1])
                right = self.compile_expr(expr[2])
                if left.type == ir.IntType(32):
                    return self.builder.mul(left, right, name="multmp")
                elif left.type == ir.DoubleType():
                    return self.builder.fmul(left, right, name="multmp")
                else:
                    raise NotImplementedError(f"Multiplication not implemented for type: {left.type}")
            elif tag == '/':
                left = self.compile_expr(expr[1])
                right = self.compile_expr(expr[2])
                if left.type == ir.IntType(32):
                    return self.builder.sdiv(left, right, name="divtmp")
                elif left.type == ir.DoubleType():
                    return self.builder.fdiv(left, right, name="divtmp")
                else:
                    raise NotImplementedError(f"Division not implemented for type: {left.type}")
            elif tag == 'call':
                func_name = expr[1]
                args_exprs = expr[2]
                llvm_function = self.module.globals.get(func_name)
                if llvm_function is None:
                    raise NameError(f"Undefined function: {func_name}")
                compiled_args = [self.compile_expr(arg) for arg in args_exprs]
                return self.builder.call(llvm_function, compiled_args, name=f"{func_name}_call")
            else:
                raise NotImplementedError(f"Unknown expression: {expr}")
        else:
            raise NotImplementedError(f"Expression compilation not implemented for: {expr}")

    def get_printf(self) -> ir.Function:
        if 'printf' in self.module.globals:
            return self.module.globals['printf']

        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        printf_func = ir.Function(self.module, printf_ty, name="printf")

        self.printf_func = printf_func
        return printf_func

    def llvm_type(self, type_token: TokenType) -> ir.Type:
        mapping = {
            TokenType.INT: ir.IntType(32),
            TokenType.FLOAT: ir.DoubleType(),
        }
        return mapping.get(type_token, ir.IntType(32))

    def get_or_create_global_string(self, string: str, name: str) -> ir.Constant:
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
        description="Optimized compiler for a simple language with import/export support"
    )
    arg_parser.add_argument('--entry', '-e', type=str, default="import.qt",
                            help="Path to the main source file")
    arg_parser.add_argument('--out_dir', '-o', type=str, default=".",
                            help="Output directory for IR and binary")
    args = arg_parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read input file
    if os.path.isfile(args.entry):
        with open(args.entry, "r") as f:
            code = f.read()
        main_filepath = os.path.abspath(args.entry)
    else:
        code = args.entry
        main_filepath = None

    # Initialize compiler components
    lexer = Lexer()
    tokens = lexer.tokenize(code)
    parser = Parser(tokens)
    main_ast = parser.program()

    # Separate imports from other nodes
    import_nodes = []
    remaining_nodes = []
    for node in main_ast:
        (import_nodes if isinstance(node, tuple) and node[0] == 'import'
         else remaining_nodes).append(node)

    # Create module and process imports
    module = ir.Module(name="module")
    import_init_funcs = []

    for imp in import_nodes:
        imported_names, imported_filename = imp[1], imp[2]
        if main_filepath:
            imp_path = os.path.join(os.path.dirname(main_filepath), imported_filename)
        else:
            imp_path = imported_filename

        with open(imp_path, "r") as f:
            imp_code = f.read()

        imp_tokens = lexer.tokenize(imp_code)
        imp_parser = Parser(imp_tokens)
        imp_ast = imp_parser.program()

        imp_analyzer = SemanticAnalyzer()
        imp_analyzer.analyze(imp_ast)

        imp_compiler = Compiler(module)
        imp_base = os.path.splitext(os.path.basename(imported_filename))[0]
        init_fn = imp_compiler.compile_globals(imp_ast, f"init_{imp_base}")
        import_init_funcs.append(init_fn)

    # Compile main module
    main_compiler = Compiler(module)
    main_init_fn = main_compiler.compile_globals(remaining_nodes, "init_main")

    # Generate main wrapper
    wrapper_main = ir.Function(module, ir.FunctionType(ir.IntType(32), []), name="main")
    main_block = wrapper_main.append_basic_block("entry")
    wrapper_builder = ir.IRBuilder(main_block)

    for fn in import_init_funcs:
        wrapper_builder.call(fn, [])
    wrapper_builder.call(main_init_fn, [])
    wrapper_builder.ret(ir.Constant(ir.IntType(32), 0))

    # Write output
    ir_file = os.path.join(args.out_dir, "output.ll")
    with open(ir_file, "w") as f:
        f.write(str(module))

    # Compile to binary
    out_binary = os.path.join(args.out_dir, "output")
    subprocess.run(["clang", "-O3", "-o", out_binary, ir_file], check=True)
    print(f"Compilation complete. Binary generated at: {out_binary}")


if __name__ == "__main__":
    main()
