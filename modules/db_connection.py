from flask import Flask, request, jsonify
import mysql.connector

app = Flask(__name__)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="meetease"
)

@app.route('/users', methods=['POST'])
def add_user():
    data = request.json
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", (data['name'], data['email']))
    conn.commit()
    return jsonify({"status": "User added"}), 201

@app.route('/users', methods=['GET'])
def get_users():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    return jsonify(cursor.fetchall())

if __name__ == '__main__':
    app.run(debug=True)
