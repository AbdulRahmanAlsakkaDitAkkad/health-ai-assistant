def test_predict_risk_changes_with_family_history(client):
    user_id = "phase9-model"

    # Add family history
    r1 = client.post("/add-family-history", json={
        "user_id": user_id,
        "items": [
            {"condition": "diabetes", "relation": "father", "notes": "type 2"}
        ]
    })
    assert r1.status_code == 200

    # Predict risk
    r2 = client.post("/predict-risk", json={
        "user_id": user_id,
        "conditions": ["diabetes"]
    })
    assert r2.status_code == 200
    preds = r2.json()["predictions"]
    assert preds[0]["risk_score"] >= 0.4
