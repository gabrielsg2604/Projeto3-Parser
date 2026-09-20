from __future__ import annotations

from collections.abc import Sequence

from Lexer import Token, TokenKind
from ast_nodes import (
    Block,
    Expr,
    FunctionDecl,
    Node,
    Parameter,
    PrintItem,
    Program,
    SourceSpan,
    Stmt,
    StringLiteral,
    TypeName,
    Assignment,
    CallExpr,
    CallStmt,
    IdentifierExpr,
    VarDecl,
    BinaryExpr,
    BinaryOperator,
    BoolLiteral,
    IfStmt,
    IntLiteral,
    PrintStmt,
    ReturnStmt,
    UnaryExpr,
    UnaryOperator,
    WhileStmt,
)


TYPE_START = {TokenKind.KW_INT, TokenKind.KW_BOOL, TokenKind.KW_VOID}
EXPRESSION_START = {
    TokenKind.IDENTIFIER,
    TokenKind.INT_LITERAL,
    TokenKind.KW_FALSE,
    TokenKind.KW_TRUE,
    TokenKind.LEFT_PAREN,
    TokenKind.LOGICAL_NOT,
    TokenKind.MINUS,
}
STATEMENT_START = TYPE_START | {
    TokenKind.IDENTIFIER,
    TokenKind.KW_IF,
    TokenKind.KW_WHILE,
    TokenKind.KW_RETURN,
    TokenKind.KW_PRINT,
    TokenKind.LEFT_BRACE,
}


TYPE_BY_TOKEN = {
    TokenKind.KW_INT: TypeName.INT,
    TokenKind.KW_BOOL: TypeName.BOOL,
    TokenKind.KW_VOID: TypeName.VOID,
}

EQUALITY_OPERATORS = {
    TokenKind.EQUAL_EQUAL: BinaryOperator.EQUAL,
    TokenKind.NOT_EQUAL: BinaryOperator.NOT_EQUAL,
}
RELATIONAL_OPERATORS = {
    TokenKind.LESS: BinaryOperator.LESS,
    TokenKind.LESS_EQUAL: BinaryOperator.LESS_EQUAL,
    TokenKind.GREATER: BinaryOperator.GREATER,
    TokenKind.GREATER_EQUAL: BinaryOperator.GREATER_EQUAL,
}
ADDITIVE_OPERATORS = {
    TokenKind.PLUS: BinaryOperator.ADD,
    TokenKind.MINUS: BinaryOperator.SUBTRACT,
}
MULTIPLICATIVE_OPERATORS = {
    TokenKind.STAR: BinaryOperator.MULTIPLY,
    TokenKind.SLASH: BinaryOperator.DIVIDE,
    TokenKind.PERCENT: BinaryOperator.REMAINDER,
}
UNARY_OPERATORS = {
    TokenKind.LOGICAL_NOT: UnaryOperator.NOT,
    TokenKind.MINUS: UnaryOperator.NEGATE,
}


class ParserError(Exception):
    def __init__(self, token: Token, expected: set[TokenKind]):
        self.token = token
        self.expected = frozenset(expected)
        super().__init__()

    @property
    def line(self) -> int:
        return self.token.line

    @property
    def column(self) -> int:
        return self.token.column

    def __str__(self) -> str:
        names = ", ".join(kind.name for kind in sorted(
            self.expected,
            key=lambda kind: kind.value,
        ))
        return (
            f"erro sintático em {self.line}:{self.column}: esperado {{{names}}}, "
            f"encontrado {self.token.kind.name} ({self.token.lexeme!r})"
        )


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        if not self.tokens:
            raise ValueError("a sequência de tokens deve terminar em EOF")
        if self.tokens[-1].kind is not TokenKind.EOF:
            raise ValueError("o último token deve ser EOF")
        if any(token.kind is TokenKind.EOF for token in self.tokens[:-1]):
            raise ValueError("EOF deve aparecer uma única vez, no final")
        self.current = 0

    def peek(self, offset: int = 0) -> Token:
        index = min(self.current + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def check(self, kind: TokenKind) -> bool:
        return self.peek().kind is kind

    def advance(self) -> Token:
        token = self.peek()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kinds: TokenKind | set[TokenKind]) -> Token:
        expected = kinds if isinstance(kinds, set) else {kinds}
        token = self.peek()
        if token.kind not in expected:
            raise ParserError(token, set(expected))
        return self.advance()

    @staticmethod
    def _token_span(token: Token) -> SourceSpan:
        return SourceSpan(
            token.line,
            token.column,
            token.line,
            token.column + len(token.lexeme),
        )

    @staticmethod
    def _start(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.start_line, value.span.start_column
        return value.line, value.column

    @staticmethod
    def _end(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.end_line, value.span.end_column
        return value.line, value.column + len(value.lexeme)

    @classmethod
    def _span(cls, first: Token | Node, last: Token | Node) -> SourceSpan:
        start_line, start_column = cls._start(first)
        end_line, end_column = cls._end(last)
        return SourceSpan(start_line, start_column, end_line, end_column)

    def parse(self) -> Program:
        return self.parse_program()

    # program ::= function* EOF
    def parse_program(self) -> Program:
        start = self.peek()
        functions: list[FunctionDecl] = []
        while self.peek().kind in TYPE_START:
            functions.append(self.parse_function())
        eof = self.expect(TokenKind.EOF)
        return Program(functions, span=self._span(start, eof))

    # function ::= type IDENTIFIER ... block
    def parse_function(self) -> FunctionDecl:
        start = self.peek()
        return_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        self.expect(TokenKind.LEFT_PAREN)
        parameters = (
            self.parse_parameter_list()
            if self.peek().kind in TYPE_START
            else []
        )
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return FunctionDecl(
            return_type,
            name.lexeme,
            parameters,
            body,
            span=self._span(start, body),
        )

    # type ::= KW_INT | KW_BOOL | KW_VOID
    def parse_type(self) -> TypeName:
        token = self.expect(TYPE_START)
        return TYPE_BY_TOKEN[token.kind]

    def parse_parameter_list(self) -> list[Parameter]:
        parameters = [self.parse_parameter()]

        while self.match(TokenKind.COMMA):    # match ja consome a virgula se bater
            parameters.append(self.parse_parameter())

        return parameters

    def parse_parameter(self) -> Parameter:
        start = self.peek()                   
        type_name = self.parse_type()          
        name = self.expect(TokenKind.IDENTIFIER)

        return Parameter(type_name, name.lexeme, span=self._span(start, name))

    def parse_block(self) -> Block:
        start = self.expect(TokenKind.LEFT_BRACE)        # { abre o span

        statements = []
        # statement* -> laco while: continua enquanto o token atual pode INICIAR um comando
        while self.peek().kind in STATEMENT_START:
            statements.append(self.parse_statement())

        end = self.expect(TokenKind.RIGHT_BRACE)         # } fecha o span
        
        return Block(statements, span=self._span(start, end))

    def parse_statement(self) -> Stmt:
        kind = self.peek().kind

        if kind in TYPE_START:                         
            return self.parse_declaration()
        if kind is TokenKind.IDENTIFIER:
            return self.parse_id_or_call_statement()
        if kind is TokenKind.KW_IF:
            return self.parse_if_statement()
        if kind is TokenKind.KW_WHILE:
            return self.parse_while_statement()
        if kind is TokenKind.KW_RETURN:
            return self.parse_return_statement()
        if kind is TokenKind.KW_PRINT:
            return self.parse_print_statement()
        if kind is TokenKind.LEFT_BRACE:                
            return self.parse_block()

        # nenhum inicio de comando bateu: o "esperado" e qualquer token de STATEMENT_START
        raise ParserError(self.peek(), STATEMENT_START)

    def parse_id_or_call_statement(self) -> Stmt:
        name = self.expect(TokenKind.IDENTIFIER)

        if self.match(TokenKind.ASSIGN):
            target = IdentifierExpr(name.lexeme, span=self._token_span(name))
            value = self.parse_expression()
            end = self.expect(TokenKind.SEMICOLON)
            return Assignment(target, value, span=self._span(name, end))

        if self.match(TokenKind.LEFT_PAREN):
            arguments = self.parse_arguments()
            rparen = self.expect(TokenKind.RIGHT_PAREN)
            call = CallExpr(name.lexeme, arguments, span=self._span(name, rparen))
            end = self.expect(TokenKind.SEMICOLON)
            return CallStmt(call, span=self._span(name, end))

        raise ParserError(self.peek(), {TokenKind.ASSIGN, TokenKind.LEFT_PAREN})

    def parse_declaration(self) -> Stmt:
        start = self.peek()
        type_name = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)

        initializer = None                               
        if self.match(TokenKind.ASSIGN):
            initializer = self.parse_expression()

        end = self.expect(TokenKind.SEMICOLON)           # o span vai ate o ";"
        return VarDecl(type_name, name.lexeme, initializer, span=self._span(start, end))

    def parse_if_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_IF)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        then_block = self.parse_block()                  

        else_block = None                                
        if self.match(TokenKind.KW_ELSE):
            else_block = self.parse_block()

        last = else_block if else_block is not None else then_block
        return IfStmt(condition, then_block, else_block, span=self._span(start, last))

    def parse_while_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_WHILE)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return WhileStmt(condition, body, span=self._span(start, body))

    def parse_return_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_RETURN)

        value = None
        if self.peek().kind in EXPRESSION_START:
            value = self.parse_expression()

        end = self.expect(TokenKind.SEMICOLON)
        return ReturnStmt(value, span=self._span(start, end))

    def parse_print_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_PRINT)
        self.expect(TokenKind.LEFT_PAREN)

        items = [self.parse_print_item()]                
        while self.match(TokenKind.COMMA):
            items.append(self.parse_print_item())

        self.expect(TokenKind.RIGHT_PAREN)
        end = self.expect(TokenKind.SEMICOLON)
        return PrintStmt(items, span=self._span(start, end))

    def parse_print_item(self) -> PrintItem:
        if self.check(TokenKind.STRING_LITERAL):
            return self.parse_string_literals()
        if self.peek().kind in EXPRESSION_START:
            return self.parse_expression()
        raise ParserError(self.peek(), EXPRESSION_START | {TokenKind.STRING_LITERAL})

    def parse_string_literals(self) -> StringLiteral:
        first = self.expect(TokenKind.STRING_LITERAL)    
        last = first
        parts = [first.value]                            

        while self.check(TokenKind.STRING_LITERAL):      
            last = self.advance()
            parts.append(last.value)

        return StringLiteral("".join(parts), span=self._span(first, last))

    def parse_expression(self) -> Expr:
        return self.parse_logical_or()

    def parse_logical_or(self) -> Expr:
        left = self.parse_logical_and()          # nivel de maior precedencia        

        while self.match(TokenKind.LOGICAL_OR):          
            right = self.parse_logical_and()
            left = BinaryExpr(
                BinaryOperator.LOGICAL_OR, left, right, span=self._span(left, right)
            )

        return left

    def parse_logical_and(self) -> Expr:
        left = self.parse_equality()                     # proximo nivel de maior precedencia

        while self.match(TokenKind.LOGICAL_AND):
            right = self.parse_equality()
            left = BinaryExpr(
                BinaryOperator.LOGICAL_AND, left, right, span=self._span(left, right)
            )

        return left

    def parse_equality(self) -> Expr:
        left = self.parse_relational()

        while self.peek().kind in EQUALITY_OPERATORS:
            operator = EQUALITY_OPERATORS[self.advance().kind]
            right = self.parse_relational()
            left = BinaryExpr(operator, left, right, span=self._span(left, right))

        return left

    def parse_relational(self) -> Expr:
        left = self.parse_additive()

        while self.peek().kind in RELATIONAL_OPERATORS:
            operator = RELATIONAL_OPERATORS[self.advance().kind]
            right = self.parse_additive()
            left = BinaryExpr(operator, left, right, span=self._span(left, right))

        return left

    def parse_additive(self) -> Expr:
        left = self.parse_multiplicative()

        while self.peek().kind in ADDITIVE_OPERATORS:
            operator = ADDITIVE_OPERATORS[self.advance().kind]
            right = self.parse_multiplicative()
            left = BinaryExpr(operator, left, right, span=self._span(left, right))

        return left

    def parse_multiplicative(self) -> Expr:
        left = self.parse_unary()

        while self.peek().kind in MULTIPLICATIVE_OPERATORS:
            operator = MULTIPLICATIVE_OPERATORS[self.advance().kind]
            right = self.parse_unary()
            left = BinaryExpr(operator, left, right, span=self._span(left, right))

        return left

    def parse_unary(self) -> Expr:
        if self.peek().kind in UNARY_OPERATORS:
            token = self.advance()
            operand = self.parse_unary()
            return UnaryExpr(
                UNARY_OPERATORS[token.kind], operand, span=self._span(token, operand)
            )

        return self.parse_primary()

    def parse_primary(self) -> Expr:
        token = self.peek()

        if token.kind is TokenKind.LEFT_PAREN:
            lparen = self.advance()
            expression = self.parse_expression()
            rparen = self.expect(TokenKind.RIGHT_PAREN)
            expression.span = self._span(lparen, rparen)
            return expression

        if token.kind is TokenKind.IDENTIFIER:
            name = self.advance()
            if self.match(TokenKind.LEFT_PAREN):        
                arguments = self.parse_arguments()
                rparen = self.expect(TokenKind.RIGHT_PAREN)
                return CallExpr(name.lexeme, arguments, span=self._span(name, rparen))
            return IdentifierExpr(name.lexeme, span=self._token_span(name))

        if token.kind is TokenKind.INT_LITERAL:
            self.advance()
            return IntLiteral(token.value, span=self._token_span(token)) 

        if token.kind is TokenKind.KW_TRUE:
            self.advance()
            return BoolLiteral(True, span=self._token_span(token))

        if token.kind is TokenKind.KW_FALSE:
            self.advance()
            return BoolLiteral(False, span=self._token_span(token))

        raise ParserError(token, EXPRESSION_START)

    def parse_arguments(self) -> list[Expr]:
        arguments = []

        if self.peek().kind in EXPRESSION_START:
            arguments.append(self.parse_expression())
            while self.match(TokenKind.COMMA):
                arguments.append(self.parse_expression()) 

        return arguments

