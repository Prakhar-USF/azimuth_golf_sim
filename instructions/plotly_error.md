We are currently having inconsistent performance of the plotly function on local device and on AWS EC2 instance. 

The current code on GitHub works perfectly on EC2 server.  

If you see some kind of `IndexError` during trajectory prediction, please try either version below and see which one works. We will keep looking into this problem and try to solve it in future updates.  
<br>

**Version 1:**  

In `./website/make_traj.py`, line 593-593:  

```
...

aspectratio=dict(x=1.55, y=1.55*abs(track_small[1][-1]/track_small[0][-1])[0],
                 z=1.55*abs(track_small[2].max()/track_small[0][-1])[0]),
```

In `./website/routes.py`, line 89:  

```
...

def predict(factors: dict, gif_name: str, is_auth=False) -> tuple:
    ...
    return x1[0], y[0], x2[0]
```

<br>

**Version 2:**  

In `./website/make_traj.py`, line 593-593:  

```
...

aspectratio=dict(x=1.55, y=1.55*abs(track_small[1][-1]/track_small[0][-1]),
                 z=1.55*abs(track_small[2].max()/track_small[0][-1])),
```

In `./website/routes.py`, line 89:  

```
...

def predict(factors: dict, gif_name: str, is_auth=False) -> tuple:
    ...
    return x1[0], y, x2
```
