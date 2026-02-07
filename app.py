import os
import uuid
import logging
from functools import wraps

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)

import torch
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# BASIC CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# APP INIT
# =========================
app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_A_SECRET_KEY"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# DUMMY USER DATABASE
# (Replace with real DB later)
# =========================
USERS = {}

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join(MODEL_FOLDER, "classifier.pt")
CLASS_NAMES = ["Class A", "Class B", "Class C"]

device = "cpu"

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None


# =========================
# HELPERS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# ---------- AUTH ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USERS:
            flash("User already exists", "danger")
            return redirect(url_for("register"))

        USERS[username] = {
            "password": generate_password_hash(password)
        }

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = USERS.get(username)

        if not user or not check_password_hash(user["password"], password):
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))

        session["user"] = username
        flash("Login successful", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))


# ---------- DASHBOARD ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])


# ---------- PROFILE ----------
@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=session["user"])


# ---------- PREDICTION ----------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("Model not available", "danger")
        return redirect(url_for("dashboard"))

    if "file" not in request.files:
        flash("No file uploaded", "warning")
        return redirect(url_for("dashboard"))

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file", "warning")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type", "danger")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    # -------- MODEL INFERENCE (DUMMY) --------
    # Replace with real preprocessing
    predicted_class = CLASS_NAMES[0]
    confidence = 0.92

    return render_template(
        "result.html",
        image_path=f"uploads/{unique_name}",
        prediction=predicted_class,
        confidence=confidence
    )


# =========================
# ERROR HANDLERS
# =========================
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500



