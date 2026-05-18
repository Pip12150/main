// init.js
console.log("📦 init.js loaded (definitions:toast)");

// -----------------------------
// Toast
// -----------------------------
toast = function (msg) {
        const div = document.createElement("div");
        div.style = "position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:black;color:white;padding:6px 12px;border-radius:6px;z-index:99999;";
        div.innerText = msg;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), 1500);
};

js_log = function (msg) {
        const box = document.querySelector("#js_log_box textarea");
        if (!box) return;

        const t = new Date().toLocaleTimeString();
        box.value += `[${t}] ${msg}\n`;
        box.scrollTop = box.scrollHeight;
}


