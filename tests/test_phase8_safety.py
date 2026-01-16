def test_phase8_safety_urgent(client):
    r = client.post("/analyze-mood", json={
        "user_id": "phase9-safety",
        "text": "I feel like I can't go on and I want to end my life."
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["analysis"]["safety_flag"] == "urgent"
    assert "safety_message" in data
    assert isinstance(data["safety_message"], str)
    assert len(data["safety_message"]) > 20
