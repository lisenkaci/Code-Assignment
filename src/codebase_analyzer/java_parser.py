"""Java structure extraction using pure-Python javalang plus Lizard metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, cast

import javalang  # type: ignore[import-untyped]
import lizard  # type: ignore[import-untyped]

from codebase_analyzer.parsed_models import (
    ParsedJavaFile,
    ParsedMethod,
    ParsedParameter,
    ParsedType,
)
from codebase_analyzer.scanner import FileRecord
from codebase_analyzer.schemas import TypeKind

TYPE_KIND_BY_CLASS_NAME = {
    "AnnotationDeclaration": TypeKind.ANNOTATION,
    "ClassDeclaration": TypeKind.CLASS,
    "EnumDeclaration": TypeKind.ENUM,
    "InterfaceDeclaration": TypeKind.INTERFACE,
}
MODIFIER_ORDER = (
    "public",
    "protected",
    "private",
    "abstract",
    "static",
    "final",
    "sealed",
    "non-sealed",
    "synchronized",
    "native",
    "strictfp",
    "default",
    "transient",
    "volatile",
)


class LizardFunction(Protocol):
    name: str
    long_name: str
    cyclomatic_complexity: int
    nloc: int
    parameter_count: int
    start_line: int
    end_line: int


class JavaParser:
    """Extract Java declarations while tolerating unsupported newer syntax."""

    def parse_file(self, repository_root: Path, record: FileRecord) -> ParsedJavaFile:
        if Path(record.path).suffix.lower() != ".java":
            raise ValueError(f"JavaParser only accepts .java files: {record.path}")
        data = (repository_root.expanduser().resolve() / record.path).read_bytes()
        return self.parse_source(record.path, data, record.sha256)

    def parse_source(self, path: str, data: bytes, sha256: str) -> ParsedJavaFile:
        source = data.decode("utf-8", errors="replace")
        complexity_functions = self._complexity_functions(path, source)
        try:
            unit = javalang.parse.parse(source)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
            return self._fallback_parse(path, source, sha256, complexity_functions)

        package = unit.package.name if unit.package is not None else None
        imports = tuple(
            f"{'static ' if item.static else ''}{item.path}{'.*' if item.wildcard else ''}"
            for item in unit.imports
        )
        types: list[ParsedType] = []
        lines = source.splitlines(keepends=True)
        for declaration in unit.types:
            types.extend(
                self._parse_type_tree(
                    declaration,
                    source=source,
                    lines=lines,
                    package=package,
                    parent_type_names=(),
                    complexity_functions=complexity_functions,
                )
            )

        return ParsedJavaFile(
            path=path,
            package=package,
            imports=imports,
            types=tuple(types),
            source=source,
            sha256=sha256,
            has_parser_warnings=False,
        )

    def _parse_type_tree(
        self,
        declaration: Any,
        *,
        source: str,
        lines: list[str],
        package: str | None,
        parent_type_names: tuple[str, ...],
        complexity_functions: tuple[LizardFunction, ...],
    ) -> list[ParsedType]:
        name = cast(str, declaration.name)
        type_names = (*parent_type_names, name)
        qualified_name = ".".join(part for part in (package, *type_names) if part)
        start_line = self._position_line(declaration)
        end_line = self._declaration_end_line(source, start_line)
        members = self._members(declaration)
        method_classes = (
            javalang.tree.AnnotationMethod,
            javalang.tree.ConstructorDeclaration,
            javalang.tree.MethodDeclaration,
        )

        methods = tuple(
            self._parse_method(
                member,
                lines=lines,
                qualified_type_name=qualified_name,
                complexity_functions=complexity_functions,
            )
            for member in members
            if isinstance(member, method_classes)
        )
        extends, implements = self._relationships(declaration)
        parsed_type = ParsedType(
            name=name,
            qualified_name=qualified_name,
            kind=TYPE_KIND_BY_CLASS_NAME.get(type(declaration).__name__, TypeKind.CLASS),
            annotations=self._annotations(declaration),
            extends=extends,
            implements=implements,
            start_line=start_line,
            end_line=end_line,
            methods=methods,
        )

        result = [parsed_type]
        for member in members:
            if type(member).__name__ in TYPE_KIND_BY_CLASS_NAME:
                result.extend(
                    self._parse_type_tree(
                        member,
                        source=source,
                        lines=lines,
                        package=package,
                        parent_type_names=type_names,
                        complexity_functions=complexity_functions,
                    )
                )
        return result

    def _parse_method(
        self,
        declaration: Any,
        *,
        lines: list[str],
        qualified_type_name: str,
        complexity_functions: tuple[LizardFunction, ...],
    ) -> ParsedMethod:
        name = cast(str, declaration.name)
        start_line = self._position_line(declaration)
        complexity = self._matching_complexity(
            name,
            start_line,
            complexity_functions,
        )
        end_line = (
            complexity.end_line
            if complexity is not None
            else self._declaration_end_line("".join(lines), start_line)
        )
        source = "".join(lines[start_line - 1 : end_line])
        signature = self._extract_signature(source)
        return_type = None
        if isinstance(declaration, javalang.tree.MethodDeclaration):
            return_type = self._type_to_string(declaration.return_type) or "void"
        elif isinstance(declaration, javalang.tree.AnnotationMethod):
            return_type = self._type_to_string(declaration.return_type)

        parameters = tuple(self._parameter(parameter) for parameter in declaration.parameters)
        return ParsedMethod(
            name=name,
            qualified_name=f"{qualified_type_name}.{name}",
            signature=signature,
            return_type=return_type,
            parameters=parameters,
            modifiers=self._modifiers(declaration),
            annotations=self._annotations(declaration),
            start_line=start_line,
            end_line=end_line,
            source=source,
            lines_of_code=max(complexity.nloc, 0) if complexity else end_line - start_line + 1,
            cyclomatic_complexity=max(complexity.cyclomatic_complexity, 1)
            if complexity
            else 1,
        )

    def _fallback_parse(
        self,
        path: str,
        source: str,
        sha256: str,
        complexity_functions: tuple[LizardFunction, ...],
    ) -> ParsedJavaFile:
        package_match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", source)
        package = package_match.group(1) if package_match else None
        imports = tuple(
            match.group(1).strip()
            for match in re.finditer(r"(?m)^\s*import\s+(.+?)\s*;", source)
        )
        declaration_match = re.search(
            r"\b(?P<kind>class|interface|enum|record|@interface)\s+(?P<name>\w+)",
            source,
        )
        if declaration_match is None:
            return ParsedJavaFile(
                path=path,
                package=package,
                imports=imports,
                types=(),
                source=source,
                sha256=sha256,
                has_parser_warnings=True,
            )

        name = declaration_match.group("name")
        qualified_name = ".".join(part for part in (package, name) if part)
        lines = source.splitlines(keepends=True)
        methods = tuple(
            self._method_from_lizard(function, lines, qualified_name)
            for function in complexity_functions
        )
        kind_text = declaration_match.group("kind")
        kind = {
            "interface": TypeKind.INTERFACE,
            "enum": TypeKind.ENUM,
            "record": TypeKind.RECORD,
            "@interface": TypeKind.ANNOTATION,
        }.get(kind_text, TypeKind.CLASS)
        parsed_type = ParsedType(
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            annotations=(),
            extends=(),
            implements=(),
            start_line=source[: declaration_match.start()].count("\n") + 1,
            end_line=max(len(lines), 1),
            methods=methods,
        )
        return ParsedJavaFile(
            path=path,
            package=package,
            imports=imports,
            types=(parsed_type,),
            source=source,
            sha256=sha256,
            has_parser_warnings=True,
        )

    def _method_from_lizard(
        self,
        function: LizardFunction,
        lines: list[str],
        qualified_type_name: str,
    ) -> ParsedMethod:
        name = function.name.split("::")[-1]
        source = "".join(lines[function.start_line - 1 : function.end_line])
        signature = self._extract_signature(source)
        return ParsedMethod(
            name=name,
            qualified_name=f"{qualified_type_name}.{name}",
            signature=signature,
            return_type=self._return_type_from_signature(signature, name),
            parameters=(),
            modifiers=(),
            annotations=(),
            start_line=function.start_line,
            end_line=function.end_line,
            source=source,
            lines_of_code=max(function.nloc, 0),
            cyclomatic_complexity=max(function.cyclomatic_complexity, 1),
        )

    @staticmethod
    def _members(declaration: Any) -> list[Any]:
        body = declaration.body
        if hasattr(body, "declarations"):
            return cast(list[Any], body.declarations)
        return cast(list[Any], body)

    def _relationships(self, declaration: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        extends_value = getattr(declaration, "extends", None)
        if extends_value is None:
            extends: tuple[str, ...] = ()
        elif isinstance(extends_value, list):
            extends = tuple(self._type_to_string(item) for item in extends_value)
        else:
            extends = (self._type_to_string(extends_value),)
        implements = tuple(
            self._type_to_string(item) for item in getattr(declaration, "implements", None) or ()
        )
        return extends, implements

    def _parameter(self, parameter: Any) -> ParsedParameter:
        parameter_type = self._type_to_string(parameter.type)
        if getattr(parameter, "varargs", False):
            parameter_type += "..."
        declaration = f"{parameter_type} {parameter.name}".strip()
        return ParsedParameter(
            name=cast(str, parameter.name),
            type=parameter_type,
            declaration=declaration,
        )

    @staticmethod
    def _type_to_string(type_node: Any) -> str:
        if type_node is None:
            return ""
        name = str(type_node.name)
        arguments = getattr(type_node, "arguments", None) or ()
        if arguments:
            rendered_arguments = []
            for argument in arguments:
                argument_type = getattr(argument, "type", None)
                if argument_type is not None:
                    rendered_arguments.append(JavaParser._type_to_string(argument_type))
                else:
                    rendered_arguments.append("?")
            name += f"<{', '.join(rendered_arguments)}>"
        sub_type = getattr(type_node, "sub_type", None)
        if sub_type is not None:
            name += f".{JavaParser._type_to_string(sub_type)}"
        name += "[]" * len(getattr(type_node, "dimensions", None) or ())
        return name

    @staticmethod
    def _annotations(declaration: Any) -> tuple[str, ...]:
        return tuple(f"@{annotation.name}" for annotation in declaration.annotations)

    @staticmethod
    def _modifiers(declaration: Any) -> tuple[str, ...]:
        modifiers = declaration.modifiers
        return tuple(modifier for modifier in MODIFIER_ORDER if modifier in modifiers)

    @staticmethod
    def _position_line(declaration: Any) -> int:
        position = declaration.position
        return cast(int, position.line) if position is not None else 1

    @staticmethod
    def _extract_signature(source: str) -> str:
        quote: str | None = None
        escaped = False
        parentheses = 0
        for index, character in enumerate(source):
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'"}:
                quote = character
            elif character == "(":
                parentheses += 1
            elif character == ")":
                parentheses = max(parentheses - 1, 0)
            elif parentheses == 0 and character in {"{", ";"}:
                return " ".join(source[:index].split())
        return " ".join(source.split())

    @staticmethod
    def _declaration_end_line(source: str, start_line: int) -> int:
        offset = sum(len(line) for line in source.splitlines(keepends=True)[: start_line - 1])
        depth = 0
        body_started = False
        quote: str | None = None
        escaped = False
        line = start_line
        for character in source[offset:]:
            if character == "\n":
                line += 1
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'"}:
                quote = character
            elif character == "{":
                body_started = True
                depth += 1
            elif character == "}" and body_started:
                depth -= 1
                if depth == 0:
                    return line
            elif character == ";" and not body_started:
                return line
        return max(line, start_line)

    @staticmethod
    def _return_type_from_signature(signature: str, method_name: str) -> str | None:
        prefix = signature.split(f"{method_name}(", maxsplit=1)[0].strip()
        words = prefix.split()
        return words[-1] if words and words[-1] != method_name else None

    @staticmethod
    def _complexity_functions(path: str, source: str) -> tuple[LizardFunction, ...]:
        analysis = lizard.analyze_file.analyze_source_code(path, source)
        return tuple(cast(list[LizardFunction], analysis.function_list))

    @staticmethod
    def _matching_complexity(
        method_name: str,
        start_line: int,
        functions: tuple[LizardFunction, ...],
    ) -> LizardFunction | None:
        candidates = [
            function
            for function in functions
            if function.name.split("::")[-1] == method_name
        ]
        return min(
            candidates,
            key=lambda function: abs(function.start_line - start_line),
            default=None,
        )


def parse_java_files(
    repository_root: Path,
    records: tuple[FileRecord, ...],
) -> dict[str, ParsedJavaFile]:
    """Parse a stable sequence of Java files."""

    parser = JavaParser()
    return {
        record.path: parser.parse_file(repository_root, record)
        for record in records
    }
