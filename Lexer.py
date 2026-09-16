from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator


class TokenKind(enum.Enum):
    """Classe já implementada: nomes e números não devem ser alterados."""

    EOF = -1

    IDENTIFIER = 1
    INT_LITERAL = 2
    STRING_LITERAL = 3

    KW_INT = 10
    KW_BOOL = 11
    KW_VOID = 12
    KW_TRUE = 13
    KW_FALSE = 14
    KW_IF = 15
    KW_ELSE = 16
    KW_WHILE = 17
    KW_RETURN = 18
    KW_PRINT = 19

    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    LESS = 25
    LESS_EQUAL = 26
    GREATER = 27
    GREATER_EQUAL = 28
    EQUAL_EQUAL = 29
    NOT_EQUAL = 30
    LOGICAL_AND = 31
    LOGICAL_OR = 32
    LOGICAL_NOT = 33
    ASSIGN = 34

    LEFT_PAREN = 40
    RIGHT_PAREN = 41
    LEFT_BRACE = 42
    RIGHT_BRACE = 43
    COMMA = 44
    SEMICOLON = 45


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: int | str | bool | None
    line: int
    column: int

    def __str__(self) -> str:
        return (
            f"<{self.kind.value}, {self.kind.name}, {self.lexeme!r}, "
            f"{self.value!r}, {self.line}, {self.column}>"
        )


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"erro léxico em {self.line}:{self.column}: {self.message}"


class Lexer:
    """Converte texto-fonte MicroC em uma sequência de tokens."""
    # Tabela de palavras reservadas da linguagem
    KEYWORDS = {
        "int": TokenKind.KW_INT,
        "bool": TokenKind.KW_BOOL,
        "void": TokenKind.KW_VOID,
        "true": TokenKind.KW_TRUE,
        "false": TokenKind.KW_FALSE,
        "if": TokenKind.KW_IF,
        "else": TokenKind.KW_ELSE,
        "while": TokenKind.KW_WHILE,
        "return": TokenKind.KW_RETURN,
        "print": TokenKind.KW_PRINT,
    }

    # Símbolos que sempre formam um token sozinhos (1 caractere)
    SIMPLE_SYMBOLS = {
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "*": TokenKind.STAR,
        "/": TokenKind.SLASH,
        "%": TokenKind.PERCENT,
        "(": TokenKind.LEFT_PAREN,
        ")": TokenKind.RIGHT_PAREN,
        "{": TokenKind.LEFT_BRACE,
        "}": TokenKind.RIGHT_BRACE,
        ",": TokenKind.COMMA,
        ";": TokenKind.SEMICOLON,
        ">": TokenKind.GREATER,
        "<": TokenKind.LESS,
        "!": TokenKind.LOGICAL_NOT,
        "=": TokenKind.ASSIGN,
    }
    
    # símbolos que sempre formam um token composto (2 caracteres)
    MULTI_SYMBOLS = {
        "==": TokenKind.EQUAL_EQUAL,
        "!=": TokenKind.NOT_EQUAL,
        "<=": TokenKind.LESS_EQUAL,
        ">=": TokenKind.GREATER_EQUAL,
        "&&": TokenKind.LOGICAL_AND,
        "||": TokenKind.LOGICAL_OR,
    }
    
    # Inicialização: mostra posição atual, linha atual e coluna atual.
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        
    # Olha o caracter atual sem consumir
    def peek(self) -> str:
        if self.position >= len(self.source):
            return ""
        return self.source[self.position]
    
    # Olha o caracter na posição a frente sem consumir
    def peek_next(self) -> str:
        next_position = self.position + 1
        if next_position >= len(self.source):
            return ""
        return self.source[next_position]
    
    # Consome um caracter e avança para o próximo
    def advance(self) -> str:
        char = self.source[self.position]
        self.position += 1
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
        
    # Função para pular espaços, quebras e comentários
    def skip_space_comments(self) -> None:
        while not self.position >= len(self.source):
            char = self.peek()

            if char in (" ", "\t", "\n"):
                self.advance()
                continue

            if char == "/" and self.peek_next() == "/":
                self.advance()
                self.advance()
                while not self.position >= len(self.source) and self.peek() != "\n":
                    self.advance()
                continue

            if char == "/" and self.peek_next() == "*":
                start_line = self.line
                start_column = self.column
                self.advance()
                self.advance()
                while not (self.peek() == "*" and self.peek_next() == "/"):
                    if self.position >= len(self.source):
                        raise LexerError(
                            "Comentário não finalizado.",
                            start_line,
                            start_column,
                        )
                    self.advance()
                self.advance()
                self.advance()
                continue

            break
    
    def tokens(self) -> Iterator[Token]:
        """Produza todos os tokens significativos e um único EOF ao final."""
        while True:
            self.skip_space_comments()
            
            if self.position >= len(self.source):
                yield Token(TokenKind.EOF, "", None, self.line, self.column)
                return
            
            line_start = self.line
            column_start = self.column
            position_start = self.position
            
            current = self.peek()
            
            if current.isascii() and (current.isalpha() or current == "_"):
                yield self.scan_identifier(line_start, column_start, position_start)
                continue

            if current.isascii() and current.isdigit():
                yield self.scan_number(line_start, column_start, position_start)
                continue

            if current == '"':
                yield self.scan_string(line_start, column_start, position_start)
                continue

            two_char = current + self.peek_next()
            if two_char in self.MULTI_SYMBOLS:
                kind = self.MULTI_SYMBOLS[two_char]
                self.advance()
                self.advance()
                yield Token(kind, two_char, None, line_start, column_start)
                continue

            if current in self.SIMPLE_SYMBOLS:
                kind = self.SIMPLE_SYMBOLS[current]
                self.advance()
                yield Token(kind, current, None, line_start, column_start)
                continue

            raise LexerError(f"Caracter inválido {current!r}", line_start, column_start)
        
    # Verifica se o caractere pode continuar um identificador (letra, dígito ou _), sempre ASCII    
    def scan_ascii(self, char: str) -> bool:
        return char.isascii() and (char.isalnum() or char == "_")
    
    # Reconhece um identificador ou palavra reservada, a partir do primeiro caractere já confirmado
    def scan_identifier(self, line_start: int, column_start: int, position_start: int):
        while self.scan_ascii(self.peek()):
            self.advance()

        lexeme = self.source[position_start:self.position]

        if lexeme in self.KEYWORDS:
            kind = self.KEYWORDS[lexeme]
            if kind == TokenKind.KW_TRUE:
                value = True
            elif kind == TokenKind.KW_FALSE:
                value = False
            else:
                value = None
        else:
            kind = TokenKind.IDENTIFIER
            value = lexeme

        return Token(kind, lexeme, value, line_start, column_start)
    
    # Reconhece um literal inteiro (somente dígitos decimais)
    def scan_number(self, line_start: int, column_start: int, position_start: int):
        while self.peek().isascii() and self.peek().isdigit():
            self.advance()

        lexeme = self.source[position_start:self.position]
        value = int(lexeme)  # Python int tem precisao arbitraria, cobre > 2**63-1
        return Token(TokenKind.INT_LITERAL, lexeme, value, line_start, column_start)

    # Reconhece um literal de string, decodificando as sequências de escape suportadas
    def scan_string(self, line_start: int, column_start: int, position_start: int):
        self.advance()  # consome a aspa de abertura

        decoded = []
        while True:
            if self.position >= len(self.source):
                # EOF antes de fechar: posicao reportada e a da aspa de abertura
                raise LexerError("String nao finalizada", line_start, column_start)

            char = self.peek()

            if char == '"':
                self.advance()
                break

            if char == "\n":
                # quebra de linha dentro de string: posicao e a da propria quebra
                raise LexerError("Quebra de linha", self.line, self.column)

            if not char.isascii():
                raise LexerError(f"Caractere inválido {char!r}", self.line, self.column)

            if char == "\\":
                backslash_line = self.line
                backslash_column = self.column
                self.advance()

                if self.position >= len(self.source):
                    raise LexerError("String nao finalizada", line_start, column_start)

                escape_char = self.peek()
                if escape_char == "n":
                    decoded.append("\n")
                elif escape_char == "t":
                    decoded.append("\t")
                elif escape_char == '"':
                    decoded.append('"')
                elif escape_char == "\\":
                    decoded.append("\\")
                else:
                    # posicao do erro de escape invalido e a da propria barra
                    raise LexerError(
                        "Sequencia inválida", backslash_line, backslash_column
                    )
                self.advance()
                continue

            decoded.append(char)
            self.advance()

        lexeme = self.source[position_start:self.position]
        value = "".join(decoded)
        return Token(TokenKind.STRING_LITERAL, lexeme, value, line_start, column_start)

    def scan(self) -> list[Token]:
        return list(self.tokens())

