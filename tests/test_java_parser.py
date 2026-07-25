from __future__ import annotations

import hashlib

from codebase_analyzer.java_parser import JavaParser
from codebase_analyzer.schemas import TypeKind

JAVA_SOURCE = b"""
package com.example;

import java.util.List;
import static java.util.Collections.emptyList;

@Deprecated
public class Example extends Base implements Runnable, Comparable<Example> {
    public Example() {}

    @Override
    public int compareTo(Example other) {
        if (other == null) {
            return 1;
        }
        return 0;
    }

    public List<String> values(String prefix, int limit) {
        return emptyList();
    }

    static class Nested {
        void nestedMethod() {}
    }
}
"""


def test_parse_java_structure_and_complexity() -> None:
    digest = hashlib.sha256(JAVA_SOURCE).hexdigest()

    parsed = JavaParser().parse_source("src/Example.java", JAVA_SOURCE, digest)

    assert parsed.package == "com.example"
    assert parsed.imports == (
        "java.util.List",
        "static java.util.Collections.emptyList",
    )
    assert parsed.sha256 == digest
    assert not parsed.has_parser_warnings
    assert len(parsed.types) == 2

    example = parsed.types[0]
    assert example.kind is TypeKind.CLASS
    assert example.qualified_name == "com.example.Example"
    assert example.annotations == ("@Deprecated",)
    assert example.extends == ("Base",)
    assert example.implements == ("Runnable", "Comparable<Example>")

    constructor, compare_to, values = example.methods
    assert constructor.name == "Example"
    assert constructor.return_type is None
    assert compare_to.signature.startswith("public int compareTo")
    assert compare_to.annotations == ("@Override",)
    assert compare_to.cyclomatic_complexity == 2
    assert compare_to.parameters[0].type == "Example"
    assert values.return_type == "List<String>"
    assert [parameter.name for parameter in values.parameters] == ["prefix", "limit"]

    nested = parsed.types[1]
    assert nested.qualified_name == "com.example.Example.Nested"
    assert nested.methods[0].qualified_name == "com.example.Example.Nested.nestedMethod"


def test_parse_reports_parser_warnings() -> None:
    source = b"class Broken { void run( { }"

    parsed = JavaParser().parse_source("Broken.java", source, hashlib.sha256(source).hexdigest())

    assert parsed.has_parser_warnings
