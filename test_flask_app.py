from app import create_app


def test_home_page_renders_form(monkeypatch):
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Ask the agent" in response.data
    assert b"Ask a question" in response.data


def test_question_route_displays_answer(monkeypatch):
    import app as app_module

    def fake_answer_question(question):
        return f"Answer for: {question}"

    monkeypatch.setattr(app_module, "answer_question", fake_answer_question)

    app = create_app()
    client = app.test_client()

    response = client.post(
        "/",
        data={"question": "What is this project?"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Answer for: What is this project?" in response.data
