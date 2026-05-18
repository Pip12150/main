// console.log("🔥 5_load_json.js loaded");
window.load_json = function (p) {
    console.log("🔥 load_json START", p);

    const payload = typeof p === "string" ? JSON.parse(p) : p;

    window.COLORS = payload.meta?.colors ?? [];
    window.CLASS_NAMES = payload.meta?.names ?? [];
    // ========== Receive COCO meta from Python ==========

    console.log("🔥 COLORS received:", window.COLORS);
    console.log("🔥 CLASS_NAMES received:", window.CLASS_NAMES);

    window.pts_list = [];
    const data = JSON.parse(payload.polygon_json);

    data.annotations.forEach(ann => {
        const seg = Array.isArray(ann.segmentation[0])
            ? ann.segmentation[0]
            : ann.segmentation;

        const pts = [];
        for (let i = 0; i < seg.length; i += 2) {
            pts.push({ x: seg[i], y: seg[i + 1] });
        }

        window.pts_list.push({
            pts,
            class_id: ann.class_id ?? ann.category_id
        });
    });

    console.log("✅ load_json done", window.pts_list);

    return {
        status: "ok",
        img_b64: payload.image_b64,
        pts_list,
        colors: COLORS,
        class_names: CLASS_NAMES
    };

};
