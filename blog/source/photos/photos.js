console.log('Photos js Hello World');

photo = {
    showLogin: function () {
        const loginHTML = `
            <div id="login-container">
                 <div id="login-box">
                    <h2>登录</h2>
                    <input type="text" id="username" placeholder="用户名">
                    <input type="password" id="password" placeholder="密码">
                    <button id="login-button">登录</button>
                    <p id="login-message"></p>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', loginHTML);

        document.getElementById("login-button").addEventListener("click", async() => {
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            const message = document.getElementById('login-message');
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ username, password })
                });
    
                if (response.ok) {
                    message.textContent = "登录成功，正在进入...";
                    message.style.color = 'green';
                    document.getElementById('login-box').remove();
                    document.getElementById('login-container').remove();
                    document.querySelector('.album_link_list').style.display = 'flex';
                    this.init();
                    } 
                else {
                    const errorText = await response.text();
                    message.textContent = errorText;
                    message.style.color = "red";
                    }
                    } catch (err) {
                        message.textContent = "连接服务器失败";
                        message.style.color = "red";
                }
            // if (username === "admin" && password === "123456") {
            //     document.getElementById('login-box').remove();
            //     document.getElementById('login-container').remove();
            //     document.querySelector('.album_link_list').style.display = 'flex';
            //     this.init();
            // } else {
            //     message.textContent = "用户名或密码错误！";
            //     message.style.color = "red";
            // }
        });
    },
    init: function () {
        var that = this;
        $.getJSON("album.json", function (data) {
            that.render(data);
        });
    },

    render: function (data) {
        var album_list = data['album'];
        album_list.sort((a, b) => {
            const getDateOnly = (album) => {
                const dt = album.image_info?.[0]?.Image_DateTime;
                if (!dt) return 0;
                // 提取年月日部分 "2023:08:12"，替换为 "2023-08-12"
                const dateOnly = dt.split(" ")[0].replace(/:/g, "-");
                return new Date(dateOnly).getTime(); // 转为时间戳
            };
            return getDateOnly(b) - getDateOnly(a); // 从新到旧排序
        });
        var html = "";
        var link_prefix = "/photos/";

        for (var i = 0; i < album_list.length; i++) {
            var album_info = album_list[i];
            var dir_name = album_info["directory"];
            var title = album_info["title"];
            var cover_url = album_info["image_info"]?.[0]?.url || "";
            var dateTime = album_info["image_info"]?.[0]?.Image_DateTime;
            var date = "未知日期";
            if (dateTime) {
                // 示例格式："2023:08:12 17:51:34"
                var dateParts = dateTime.split(" ")[0].split(":"); // ["2023", "08", "12"]
                if (dateParts.length === 3) {
                    date = `${dateParts[0]}-${dateParts[1]}-${dateParts[2]}`;
                }
            }
            html += `
            <div class="card">
                <a href="${link_prefix + dir_name}/">
                    <img class="cover" src="${cover_url}" alt="${title}">
                    <div class="card-title">${title}</div>
                    <div class="album-meta">${date}</div>
                </a>
            </div>`;
            }

        $(".album_link_list").append(html);
        this.minigrid();
    },

    minigrid: function () {
        var grid = new Minigrid({
            container: '.album_link_list',
            item: '.card',
            gutter: 12
        });
        grid.mount();
        $(window).resize(function () {
            grid.mount();
        });
    }
}

photo.showLogin();