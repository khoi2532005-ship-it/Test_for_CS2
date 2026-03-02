from flask import Flask, render_template, request, jsonify
from utils.detector import Detector

app = Flask(__name__)
detector = Detector()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        body = request.json
        if not body or "image" not in body:
            return jsonify({"error": "No image field in request"}), 400

        data = body["image"]
        print(f"[DEBUG] data type: {type(data)}")
        print(f"[DEBUG] data length: {len(data) if data else 0}")
        print(f"[DEBUG] data prefix: {data[:50] if data else 'EMPTY'}")

        result = detector.process_frame(data)
        return jsonify({"image": result})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
