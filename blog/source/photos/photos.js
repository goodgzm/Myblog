console.log('Photos js Hello World');

photo = {
    init: function () {
        var that = this;
        $.getJSON("album.json", function (data) {
            that.render(data);
        });
    },

    render: function (data) {
        var album_list = data['album'];
        var html = "";
        var link_prefix = "/photos/";

        for (var i = 0; i < album_list.length; i++) {
            var album_info = album_list[i];
            var dir_name = album_info["directory"];
            var title = album_info["title"];
            var cover_url = album_info["image_info"]?.[0]?.url || "";

            html += `
            <div class="card">
                <a href="${link_prefix + dir_name}/">
                    <img class="cover" src="${cover_url}" alt="${title}">
                    <div class="card-title">${title}</div>
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

photo.init();