from __future__ import annotations

from app.engines.klein_tryon_lora_engine import build_klein_tryon_prompt


def test_klein_prompt_builder_preserves_tryon_prefix():
    prompt = "TRYON full body shot with a blue jacket"
    assert build_klein_tryon_prompt(prompt) == prompt


def test_klein_prompt_builder_adds_tryon_prefix_to_user_prompt():
    assert build_klein_tryon_prompt("replace the shirt") == "TRYON replace the shirt"


def test_klein_prompt_builder_generates_default_prompt():
    prompt = build_klein_tryon_prompt(None)
    assert prompt.startswith("TRYON")
    assert "reference images" in prompt
    assert "full body shot" in prompt
