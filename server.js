const path = require('path');
const os = require('os');
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcrypt');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

const dbPath = path.join(os.homedir(), 'tool', 'users.db');
const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('无法打开数据库:', err.message);
    } else {
        console.log('数据库连接成功:', dbPath);
    }
});

// 登录接口
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    db.get("SELECT password FROM users WHERE username = ?", [username], async (err, row) => {
        if (err) return res.status(500).send("服务器错误");
        if (!row) return res.status(401).send("用户名或密码错误");

        const match = await bcrypt.compare(password, row.password);
        if (match) {
            res.send("登录成功");
        } else {
            res.status(401).send("用户名或密码错误");
        }
    });
});

app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});