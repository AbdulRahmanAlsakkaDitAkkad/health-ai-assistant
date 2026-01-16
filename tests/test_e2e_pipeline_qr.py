import base64

PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)

def test_e2e_upload_to_qr_roundtrip(client):
    user_id = "phase9-e2e"

    png_bytes = base64.b64decode(PNG_1X1_B64)
    files = {"file": ("test.png", png_bytes, "image/png")}

    r1 = client.post(f"/upload-document?user_id={user_id}", files=files)
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1["status"] == "success"
    assert "summary" in j1

    r2 = client.post("/generate-dht-qr", json={
        "user_id": user_id,
        "include_notes": False,
        "include_history": True,
        "include_family": True,
        "include_surgeries": False,
        "include_ingested_items": True,
        "include_wearables": True,
        "include_predictions": True,
        "expires_in_minutes": 10,
        "scope_label": "phase9_e2e"
    })
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["status"] == "success"
    assert isinstance(j2["token_b64"], str)

    r3 = client.post("/decode-dht-token", json={"token_b64": j2["token_b64"]})
    assert r3.status_code == 200
    j3 = r3.json()
    assert j3["status"] == "success"
    assert j3["envelope"]["user_id"] == user_id
    assert j3["envelope"]["scope"] == "phase9_e2e"
    assert j3["envelope"]["data"]["user_id"] == user_id
