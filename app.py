from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from utils.detector import Detector

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
detector = Detector()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    emit("result", detector.process_frame(data))

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
