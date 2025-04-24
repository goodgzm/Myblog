const path = require('path');
const os = require('os');
const bcrypt = require('bcrypt');
const sqlite3 = require('sqlite3').verbose();

// 创建 SQLite 数据库连接
const dbPath = path.join(os.homedir(), 'tool', 'users.db');
const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('无法打开数据库:', err.message);
    } else {
        console.log('数据库连接成功:', dbPath);
    }
});
// 加密用户密码
const username = 'gzm';
const plainPassword = '18255411765gzm';

bcrypt.hash(plainPassword, 10, (err, hashedPassword) => {
    if (err) {
        console.log("密码加密失败:", err);
        return;
    }

    // 插入加密后的密码到数据库
    db.run("INSERT INTO users (username, password) VALUES (?, ?)", [username, hashedPassword], function (err) {
        if (err) {
            console.error("插入失败:", err);
        } else {
            console.log(`用户 ${username} 已插入，ID 为 ${this.lastID}`);
        }
    });
});