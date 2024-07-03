var remote_name = "QWen2";

function date_str() {
    var d = new Date();
    var strDate = d.getFullYear() + "/" + (d.getMonth()+1) + "/" + d.getDate();
    var time = d.getHours() + ":" + d.getMinutes() + ":" + d.getSeconds();
    return strDate + " | " + time;

}

function convertHTML(str) {
    const symbols = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&apos;"
    }
    for (const symbol in symbols) {
        if (str.indexOf(symbol) >= 0) {
        const newStr = str.replaceAll(symbol, symbols[symbol])
        return newStr
        }
    }
    return str;
}

function append_myself(msg) {
    var htxt = '<div id="your-chat" class="your-chat">';
    htxt = htxt + '<p class="your-chat-balloon">' + msg + '</p>';
    htxt = htxt + '<p class="chat-datetime"><span class="glyphicon glyphicon-ok"></span>' + date_str() + '</p>';
    htxt = htxt + "</div>";

    $("#chat-area").append(htxt);
    var objDiv = document.getElementById("chat-area");
    objDiv.scrollTop = objDiv.scrollHeight;
}

function append_remote(data) {
    var htxt = '<div id="friends-chat" class="friends-chat">';

    htxt += '<div class="profile friends-chat-photo">';
    htxt += '   <img src="qwen.png" alt="">';
    htxt += '</div>';

    htxt += '<div class="friends-chat-content">';
    htxt += '   <p class="friends-chat-name">' + remote_name + '</p>';
    htxt += '   <p class="friends-chat-balloon">' + convertHTML(data) + '</p>';
    htxt += '   <h5 class="chat-datetime">' + date_str() + '</h5>';
    htxt += '</div>';

    htxt += '</div>';

    $("#chat-area").append(htxt);

    var objDiv = document.getElementById("chat-area");
    objDiv.scrollTop = objDiv.scrollHeight;
}

function init() {
    $( "#chat_btn" ).on( "click", function() {
        var msg = $("#chat_txt").val();

        append_myself(msg);

        var jqxhr = $.post("/chat", msg, function(data) {
            append_remote(data);
        })
        .done(function() {
            //alert( "second success" );
        })
        .fail(function() {
            alert( "服务器错误" );
        })
        .always(function() {
            //alert( "finished" );
        });
    } );
}
$( document ).ready(function() {
    init();
});

