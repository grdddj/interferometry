import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from fitter_data import y_data, x_data


def fit_sin(tt, yy) -> dict:
    """
    Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    # guess_amp = np.std(yy) * 2.0**0.5
    guess_amp = 3 * np.std(yy) * 2.0**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    params, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = params
    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c

    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "params": params,
        "guess": guess,
    }


USE_FOREIGN_DATA = False
if USE_FOREIGN_DATA:
    N, amp, omega, phase, offset, noise = 500, 1.0, 2.0, 0.5, 4.0, 3
    # N, amp, omega, phase, offset, noise = 50, 1., .4, .5, 4., .2
    # N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
    tt = np.linspace(0, 10, N)
    tt2 = np.linspace(0, 10, 10 * N)
    yy = amp * np.sin(omega * tt + phase) + offset
    yynoise = yy + noise * (np.random.random(len(tt)) - 0.5)

    res = fit_sin(tt, yynoise)
    print(
        "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s"
        % res
    )

    plt.plot(tt, yy, "-k", label="y", linewidth=2)
    plt.plot(tt, yynoise, "ok", label="y with noise")
    plt.plot(tt2, res["fitfunc"](tt2), "r-", label="y fit curve", linewidth=2)
    plt.legend(loc="best")
    plt.show()
else:
    res = fit_sin(x_data, y_data)
    print("res", res)

    import matplotlib.pyplot as plt

    y_data = np.array(y_data)
    x_data = np.array(x_data)

    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data)

    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(
        x_data,
        res["fitfunc"](x_data),
        label="Fitted function",
    )
    plt.title(f"{res['params']}")

    plt.legend(loc="best")

    plt.show()
