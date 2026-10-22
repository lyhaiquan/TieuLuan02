import numpy as np, torch, torch.nn as nn
import rnn_numpy as R
torch.set_default_dtype(torch.float64)
out = {}
for cell in ("RNN","GRU","LSTM"):
    m = R.SeqNetNP(cell, 3, 8, seed=1)
    X = np.random.default_rng(0).normal(size=(5,12,3)); y = (np.random.default_rng(1).random(5)>.5).astype(float)
    layer = {"RNN":nn.RNN,"GRU":nn.GRU,"LSTM":nn.LSTM}[cell](3,8,batch_first=True)
    with torch.no_grad():
        for k,v in m.rnn.p.items(): getattr(layer,k.replace("W_ih","weight_ih_l0").replace("W_hh","weight_hh_l0").replace("b_ih","bias_ih_l0").replace("b_hh","bias_hh_l0")).copy_(torch.tensor(v))
    head = nn.Linear(8,1)
    with torch.no_grad(): head.weight.copy_(torch.tensor(m.p["W_o"])); head.bias.copy_(torch.tensor(m.p["b_o"]))
    Xt = torch.tensor(X)
    o,_ = layer(Xt); logit = head(o[:,-1]).squeeze(-1)
    loss_t = nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y)); loss_t.backward()
    lg = m.forward(X); loss, d = R.bce_grad(lg, y); g = m.backward(d)
    dfwd = np.abs(lg - logit.detach().numpy()).max()
    dg = max(np.abs(g["rnn.W_ih"]-layer.weight_ih_l0.grad.numpy()).max(), np.abs(g["rnn.W_hh"]-layer.weight_hh_l0.grad.numpy()).max(),
             np.abs(g["rnn.b_ih"]-layer.bias_ih_l0.grad.numpy()).max(), np.abs(g["rnn.b_hh"]-layer.bias_hh_l0.grad.numpy()).max(),
             np.abs(g["W_o"]-head.weight.grad.numpy()).max())
    # finite differences on W_hh
    P = m.params(); k="rnn.W_hh"; idx=(2,3); eps=1e-6
    P[k][idx]+=eps; lp=R.bce_grad(m.forward(X),y)[0]; P[k][idx]-=2*eps; lm=R.bce_grad(m.forward(X),y)[0]; P[k][idx]+=eps
    fd = (lp-lm)/(2*eps); fd_err=abs(fd-g[k][idx])/max(1e-12,abs(fd)+abs(g[k][idx]))
    print(cell, "forward diff vs torch %.2e | grad diff vs torch %.2e | finite-diff rel err %.2e | params %d"%(dfwd,dg,fd_err,m.n_params()))
    out[cell]=dict(fwd_diff=float(dfwd), grad_diff=float(dg), fd_rel_err=float(fd_err), params=m.n_params())
import json; json.dump(out, open("results/numpy_verify.json","w"), indent=1)
