from scripts.evoke_plain_query import plain_text_to_raw_terms


def test_plain_text_to_raw_terms_drops_raw_query_syntax() -> None:
    text = (
        'A high microerythrocyte count raises vulnerability to severe '
        'anemia in homozygous alpha (+)- thalassemia trait subjects.'
    )

    assert plain_text_to_raw_terms(text) == (
        'a high microerythrocyte count raises vulnerability to severe '
        'anemia in homozygous alpha thalassemia trait subjects'
    )


def test_plain_text_to_raw_terms_lowercases_boolean_words() -> None:
    assert plain_text_to_raw_terms('CAT AND (BIRD OR NOT DOG)') == (
        'cat and bird or not dog'
    )
