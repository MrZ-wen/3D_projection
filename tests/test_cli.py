from __future__ import annotations

from pointcloud_projection.cli import parse_interactive_rotation_command


def test_parse_interactive_rotation_command_supports_axis_increment() -> None:
    action, values = parse_interactive_rotation_command("x 5")
    assert action == "x"
    assert values == (5.0,)


def test_parse_interactive_rotation_command_supports_set() -> None:
    action, values = parse_interactive_rotation_command("set 1 -2 3")
    assert action == "set"
    assert values == (1.0, -2.0, 3.0)


def test_parse_interactive_rotation_command_rejects_invalid_input() -> None:
    try:
        parse_interactive_rotation_command("oops")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid interactive command to raise ValueError")
