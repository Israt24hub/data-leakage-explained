/*
 * The simulation behind the page: a synthetic version of the UCI Bank Marketing data,
 * a logistic-regression model trained in the browser, and ROC-AUC.
 * Works in the browser (window.LeakSim) and in Node (module.exports) for testing.
 */
;(function (root) {
    "use strict"

    // small seeded random generator (mulberry32), so every visitor sees the same data
    function rng(seed) {
        var a = seed >>> 0
        return function () {
            a = (a + 0x6d2b79f5) >>> 0
            var t = a
            t = Math.imul(t ^ (t >>> 15), t | 1)
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296
        }
    }
    function normal(r) {
        var u = 1 - r(), v = r()
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
    }
    function pick(r, items, probs) {
        var x = r(), acc = 0
        for (var i = 0; i < items.length; i++) {
            acc += probs[i]
            if (x < acc) return items[i]
        }
        return items[items.length - 1]
    }

    var JOBS = ["admin", "blue-collar", "technician", "services", "management", "retired", "student"]
    var JOB_P = [0.22, 0.24, 0.17, 0.1, 0.17, 0.06, 0.04]
    var MONTHS = ["mar", "may", "jun", "jul", "aug", "oct", "nov"]
    var MONTH_P = [0.04, 0.34, 0.13, 0.17, 0.16, 0.05, 0.11]

    // One row per phone call. 'late' marks the last stretch of the campaign, where far more
    // people say yes (in the real data: about 52% in 2010 vs 11% overall).
    function makeData(n, seed) {
        var r = rng(seed)
        var rows = []
        for (var i = 0; i < n; i++) {
            var t = i / n
            var late = t >= 0.9 ? 1 : 0
            var job = pick(r, JOBS, JOB_P)
            var month = pick(r, MONTHS, MONTH_P)
            var age = Math.round(Math.min(90, Math.max(18, 41 + 10 * normal(r) + (job === "retired" ? 22 : 0) - (job === "student" ? 16 : 0))))
            var balance = Math.round(Math.exp(6.5 + 1.2 * normal(r)) - 300)
            var contact = pick(r, ["cellular", "telephone", "unknown"], [0.66, 0.07, 0.27])
            var campaign = 1 + Math.floor(-Math.log(1 - r()) * 1.6)
            var prevSuccess = r() < 0.07 ? 1 : 0
            var logit =
                -3.1 +
                1.8 * (job === "retired" || job === "student") +
                3.0 * prevSuccess +
                (1.6 - 2.0 * late) * (month === "mar" || month === "oct") +
                (0.9 * late - 0.8) * (month === "may") -
                0.4 * (campaign - 1) -
                (1.8 - 1.5 * late) * (contact === "unknown") +
                0.00015 * Math.min(Math.max(balance, -2000), 20000) +
                3.2 * late
            var yes = r() < 1 / (1 + Math.exp(-logit)) ? 1 : 0
            // the leak: call length depends on the answer the call produced
            var duration = Math.round(Math.exp(yes ? 5.9 + 0.55 * normal(r) : 5.3 + 0.6 * normal(r)))
            rows.push({ t: t, late: late, age: age, job: job, balance: balance, contact: contact, month: month,
                campaign: campaign, prevSuccess: prevSuccess, duration: duration, y: yes })
        }
        return rows
    }

    // the columns a visitor can switch on and off
    var FEATURES = [
        { key: "age", label: "Age", known: "before" },
        { key: "job", label: "Job", known: "before" },
        { key: "balance", label: "Account balance", known: "before" },
        { key: "contact", label: "Contact type", known: "before" },
        { key: "month", label: "Month", known: "before" },
        { key: "campaign", label: "Calls this campaign", known: "before" },
        { key: "prevSuccess", label: "Said yes last time", known: "before" },
        { key: "duration", label: "Call duration", known: "after" },
    ]

    function encodeRow(row, keys) {
        var x = []
        keys.forEach(function (k) {
            if (k === "age") x.push((row.age - 41) / 11)
            else if (k === "balance") x.push(Math.sign(row.balance) * Math.log1p(Math.abs(row.balance)) / 4 - 1.5)
            else if (k === "campaign") x.push((Math.min(row.campaign, 10) - 2.5) / 2)
            else if (k === "prevSuccess") x.push(row.prevSuccess)
            else if (k === "duration") x.push((Math.log(Math.max(row.duration, 1)) - 5.4) / 0.7)
            else if (k === "job") JOBS.forEach(function (j) { x.push(row.job === j ? 1 : 0) })
            else if (k === "month") MONTHS.forEach(function (m) { x.push(row.month === m ? 1 : 0) })
            else if (k === "contact") ["cellular", "telephone", "unknown"].forEach(function (c) { x.push(row.contact === c ? 1 : 0) })
        })
        return x
    }

    // logistic regression, full-batch gradient descent with a little L2
    function train(rows, keys) {
        var X = rows.map(function (r) { return encodeRow(r, keys) })
        var d = X.length ? X[0].length : 0
        var w = new Float64Array(d), b = 0
        var lr = 0.5, l2 = 0.001, n = X.length
        for (var it = 0; it < 300; it++) {
            var gw = new Float64Array(d), gb = 0
            for (var i = 0; i < n; i++) {
                var z = b, xi = X[i]
                for (var j = 0; j < d; j++) z += w[j] * xi[j]
                var err = 1 / (1 + Math.exp(-z)) - rows[i].y
                gb += err
                for (var k = 0; k < d; k++) gw[k] += err * xi[k]
            }
            for (var m = 0; m < d; m++) w[m] -= lr * (gw[m] / n + l2 * w[m])
            b -= lr * gb / n
        }
        return function predict(row) {
            var x = encodeRow(row, keys), z = b
            for (var j = 0; j < d; j++) z += w[j] * x[j]
            return 1 / (1 + Math.exp(-z))
        }
    }

    // ROC curve points and the area under it
    function roc(scores, labels) {
        var idx = scores.map(function (_, i) { return i }).sort(function (a, b) { return scores[b] - scores[a] })
        var P = labels.reduce(function (s, v) { return s + v }, 0), N = labels.length - P
        var tp = 0, fp = 0, pts = [[0, 0]], area = 0, prevFpr = 0, prevTpr = 0
        for (var i = 0; i < idx.length; i++) {
            if (labels[idx[i]]) tp++
            else fp++
            var last = i === idx.length - 1 || scores[idx[i + 1]] !== scores[idx[i]]
            if (last) {
                var fpr = fp / N, tpr = tp / P
                area += (fpr - prevFpr) * (tpr + prevTpr) / 2
                pts.push([fpr, tpr])
                prevFpr = fpr
                prevTpr = tpr
            }
        }
        return { auc: area, points: pts }
    }

    // random 80/20 split vs "train on the past, test on the future"
    function evaluate(rows, keys, mode, seed) {
        var train_, test_
        if (mode === "time") {
            train_ = rows.filter(function (r) { return !r.late })
            test_ = rows.filter(function (r) { return r.late })
        } else {
            var r = rng(seed || 7)
            var shuffled = rows.slice().sort(function () { return r() - 0.5 })
            var cut = Math.floor(rows.length * 0.8)
            train_ = shuffled.slice(0, cut)
            test_ = shuffled.slice(cut)
        }
        if (!keys.length) return { auc: 0.5, points: [[0, 0], [1, 1]], test: test_ }
        var predict = train(train_, keys)
        var res = roc(test_.map(predict), test_.map(function (r) { return r.y }))
        res.test = test_
        return res
    }

    var api = { makeData: makeData, FEATURES: FEATURES, evaluate: evaluate, roc: roc, train: train }
    if (typeof module !== "undefined" && module.exports) module.exports = api
    else root.LeakSim = api
})(this)
