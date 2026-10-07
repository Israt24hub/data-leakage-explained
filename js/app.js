;(function () {
    "use strict"
    var S = window.LeakSim
    var SEEDS = [3, 2, 4, 1, 5, 6]
    var seedIndex = 0
    var rows = S.makeData(4000, SEEDS[seedIndex])
    var honestKeys = S.FEATURES.filter(function (f) { return f.known === "before" }).map(function (f) { return f.key })
    var selected = honestKeys.slice()
    var cache = {}

    function result(keys, mode) {
        var id = SEEDS[seedIndex] + "|" + mode + "|" + keys.slice().sort().join(",")
        if (!cache[id]) cache[id] = S.evaluate(rows, keys, mode)
        return cache[id]
    }
    function fmt(x) { return x.toFixed(3) }

    /* ---------------- column chips ---------------- */
    var chipBox = document.getElementById("chips")
    S.FEATURES.forEach(function (f) {
        var b = document.createElement("button")
        b.type = "button"
        b.className = "chip" + (f.known === "after" ? " leaky" : "")
        b.dataset.key = f.key
        b.innerHTML = f.label + (f.known === "after" ? " <small>after the call</small>" : "")
        b.addEventListener("click", function () {
            var i = selected.indexOf(f.key)
            if (i === -1) selected.push(f.key)
            else selected.splice(i, 1)
            update()
        })
        chipBox.appendChild(b)
    })
    document.getElementById("preset-honest").addEventListener("click", function () { selected = honestKeys.slice(); update() })
    document.getElementById("preset-all").addEventListener("click", function () { selected = S.FEATURES.map(function (f) { return f.key }); update() })

    /* ---------------- ROC chart ---------------- */
    var NS = "http://www.w3.org/2000/svg"
    var svg = document.getElementById("roc")
    var M = { l: 44, r: 12, t: 12, b: 44 }, W = 360 - M.l - M.r, H = 340 - M.t - M.b
    function el(name, attrs, parent) {
        var n = document.createElementNS(NS, name)
        for (var k in attrs) n.setAttribute(k, attrs[k])
        ;(parent || svg).appendChild(n)
        return n
    }
    function sx(v) { return M.l + v * W }
    function sy(v) { return M.t + (1 - v) * H }
    el("rect", { x: M.l, y: M.t, width: W, height: H, fill: "none", class: "axis" })
    el("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), class: "axis", "stroke-dasharray": "3 4" })
    ;[0, 0.5, 1].forEach(function (v) {
        el("text", { x: sx(v), y: M.t + H + 18, "text-anchor": "middle", class: "tick" }).textContent = v
        el("text", { x: M.l - 8, y: sy(v) + 4, "text-anchor": "end", class: "tick" }).textContent = v
    })
    el("text", { x: M.l + W / 2, y: 336, "text-anchor": "middle", class: "axis-label" }).textContent = "False positive rate"
    var yl = el("text", { x: 14, y: M.t + H / 2, "text-anchor": "middle", class: "axis-label", transform: "rotate(-90 14 " + (M.t + H / 2) + ")" })
    yl.textContent = "True positive rate"
    el("text", { x: sx(0.62), y: sy(0.5), class: "tick" }).textContent = "guessing"
    var area = el("path", { class: "area" })
    var ghost = el("path", { class: "ghost" })
    var curve = el("path", { class: "curve" })
    function pathOf(points) {
        return points.map(function (p, i) { return (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1) }).join(" ")
    }

    /* ---------------- update ---------------- */
    var aucEl = document.getElementById("auc")
    var deltaEl = document.getElementById("delta")
    var ghostLabel = document.getElementById("ghost-label")

    function update() {
        Array.prototype.forEach.call(chipBox.children, function (b) {
            b.setAttribute("aria-pressed", selected.indexOf(b.dataset.key) !== -1 ? "true" : "false")
        })
        var hasLeak = selected.indexOf("duration") !== -1
        var other = hasLeak ? selected.filter(function (k) { return k !== "duration" }) : selected.concat("duration")
        var now = result(selected, "random")
        var alt = result(other, "random")
        var color = hasLeak ? "var(--leak)" : "var(--honest)"
        var altColor = hasLeak ? "var(--honest)" : "var(--leak)"

        aucEl.textContent = selected.length ? fmt(now.auc) : "0.500"
        aucEl.classList.toggle("leaky", hasLeak)
        curve.setAttribute("d", pathOf(now.points))
        curve.setAttribute("stroke", color)
        area.setAttribute("d", pathOf(now.points) + " L" + sx(1) + " " + sy(0) + " L" + sx(0) + " " + sy(0) + " Z")
        area.setAttribute("fill", color)
        ghost.setAttribute("d", pathOf(alt.points))
        ghost.setAttribute("stroke", altColor)
        document.querySelector(".key.solid").style.borderTopColor = color
        document.querySelector(".key.dashed").style.borderTopColor = altColor
        ghostLabel.textContent = hasLeak ? "same columns without call duration" : "same columns, plus call duration"

        var gain = (hasLeak ? now.auc - alt.auc : alt.auc - now.auc)
        if (hasLeak) {
            deltaEl.innerHTML = "<b>+" + fmt(gain) + "</b> from call duration, a column you won't have when you need the prediction."
        } else if (selected.length) {
            deltaEl.innerHTML = "Switching on call duration would add <b>+" + fmt(gain) + "</b>. Try it."
        } else {
            deltaEl.textContent = "Pick at least one column."
        }
    }

    /* ---------------- time section ---------------- */
    function rate(list) { return list.reduce(function (s, r) { return s + r.y }, 0) / list.length }
    function setBar(id, value, max, label) {
        document.getElementById("bar-" + id).style.width = Math.max(0, Math.min(1, value / max)) * 100 + "%"
        document.getElementById("val-" + id).textContent = label
    }
    function updateTime() {
        var random = result(honestKeys, "random")
        var time = result(honestKeys, "time")
        // bars start at 0.5 (guessing), so the difference is visible
        setBar("random", random.auc - 0.5, 0.5, fmt(random.auc))
        setBar("time", time.auc - 0.5, 0.5, fmt(time.auc))
        var early = rate(rows.filter(function (r) { return !r.late }))
        var late = rate(rows.filter(function (r) { return r.late }))
        setBar("early", early, 1, Math.round(early * 100) + "%")
        setBar("late", late, 1, Math.round(late * 100) + "%")
    }

    document.getElementById("reroll").addEventListener("click", function () {
        seedIndex = (seedIndex + 1) % SEEDS.length
        rows = S.makeData(4000, SEEDS[seedIndex])
        update()
        updateTime()
    })

    update()
    // let the first paint happen before the second set of models trains
    setTimeout(updateTime, 30)
})()
