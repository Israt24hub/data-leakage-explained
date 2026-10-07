# The Column That Fooled Every Model

**Live:** https://israt24hub.github.io/data-leakage-explained/

An interactive explainer on **data leakage**, built from our CSE445 study of the UCI Bank Marketing data. Visitors switch columns on and off, and a model trained in their browser shows how one leaky column, `duration`, makes the score jump.

## What's on the page

1. **The task:** predict, before calling, which bank customers will say yes to a term deposit.
2. **A timeline** showing when each column becomes known. Call duration is only recorded after the customer has answered.
3. **A live lab:** a logistic-regression model trains in the browser on 4,000 synthetic calls. Switching on *call duration* raises ROC-AUC by about 0.15, close to the 0.14 we measured on the real data.
4. **Our real results:** `duration` inflated ROC-AUC by about 0.14 in every model family we tried (classical ML, iLTM, DistilBERT/RoBERTa, Phi-3-mini).
5. **The time trap:** a random split vs training on the past and testing on the future, plus the jump in "yes" answers (about 11% overall vs about 52% in 2010).
6. **How to catch it**, with a link to [LeakGuard](https://github.com/Israt24hub/leakguard).

## How it works

- `js/sim.js` generates the synthetic calls with a seeded random generator, trains logistic regression with gradient descent (one-hot categories, L2 regularisation), and computes ROC curves and AUC. It also runs in Node for testing.
- `js/app.js` wires up the controls and draws the ROC curve in SVG. No libraries, no build step.
- The synthetic data is tuned to match the real study: about 11% "yes" overall, about 50% in the final period, a leakage gap near 0.14, and a past-to-future ROC-AUC near 0.65.

Run it locally by opening `index.html` in a browser.

## Credits

By Israt Zaman Srity, based on our CSE445 group project at North South University ([write-up](https://github.com/Israt24hub/bank-marketing-leakage)).
Data: Moro, Cortez & Rita (2014), *A data-driven approach to predict the success of bank telemarketing*, UCI Machine Learning Repository.
