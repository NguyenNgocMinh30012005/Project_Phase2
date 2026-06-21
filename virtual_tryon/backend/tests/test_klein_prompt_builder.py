from __future__ import annotations

from app.engines.klein_prompt_builder import build_klein_tryon_prompt


def test_klein_prompt_builder_preserves_tryon_prefix():
    prompt = "TRYON full body shot with a blue jacket"
    normalized = build_klein_tryon_prompt(None, None, None, "upper_body", extra_instruction=prompt)
    assert normalized.startswith(prompt)


def test_klein_prompt_builder_adds_tryon_prefix_to_user_prompt():
    assert build_klein_tryon_prompt(None, None, None, "upper_body", extra_instruction="replace the shirt").startswith(
        "TRYON replace the shirt"
    )


def test_klein_prompt_builder_generates_default_prompt():
    prompt = build_klein_tryon_prompt(None, None, None, "upper_body")
    assert prompt.startswith("TRYON")
    assert "blue velvet wrap V-neck short-sleeve top" in prompt
    assert "Remove the original pink sleeveless shirt entirely" in prompt
    assert "The final image is a full body shot." in prompt


def test_klein_prompt_upper_body_preserves_pants():
    prompt = build_klein_tryon_prompt(
        "model standing front-facing",
        "a red blazer",
        "black pants",
        "upper_body",
    )
    assert prompt.startswith("TRYON")
    assert "Preserve the person's face, hair, hands, body shape, black pants, pose, and background." in prompt


def test_klein_prompt_full_outfit_uses_top_and_bottom_descriptions():
    prompt = build_klein_tryon_prompt(
        "model standing front-facing",
        "a linen shirt",
        "wide-leg trousers",
        "full_outfit",
    )
    assert "a linen shirt and wide-leg trousers" in prompt
    assert "The final image is a full body shot." in prompt
