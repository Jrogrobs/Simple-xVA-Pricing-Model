import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')


class XvaPricingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("xVA Pricing Dashboard (CVA/DVA/FVA)")
        self.root.geometry("1200x920")

        # State
        self._figure = None
        self._canvas = None
        self.coupon_entry = None  # will store widget ref

        self._build_ui()

    # =========================
    # UI
    # =========================
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        for r in (2, 3, 4):
            main.rowconfigure(r, weight=1)

        header = ttk.Label(
            main,
            text="Simple xVA Model",
            font=("Segoe UI", 14, "bold"),
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # ---------- Inputs ----------
        inputs = ttk.LabelFrame(main, text="Inputs", padding=10)
        inputs.grid(row=1, column=0, sticky="ew")
        for c in range(10):
            inputs.columnconfigure(c, weight=1)

        # Trade specifics
        ttk.Label(inputs, text="Trade specifics", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 2))

        ttk.Label(inputs, text="Trade type").grid(row=1, column=0, sticky="w")
        self.trade_type = tk.StringVar(value="Payer")
        ttk.Combobox(inputs, textvariable=self.trade_type, values=["Payer", "Receiver"], width=12, state="readonly")\
            .grid(row=1, column=1, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Notional").grid(row=2, column=0, sticky="w")
        self.notional_var = tk.StringVar(value="10000000")  # 10mm
        ttk.Entry(inputs, textvariable=self.notional_var, width=14).grid(row=2, column=1, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Maturity / Tenor").grid(row=3, column=0, sticky="w")
        self.tenor_var = tk.StringVar(value="5Y")
        ttk.Combobox(
            inputs,
            textvariable=self.tenor_var,
            values=["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y", "18M", "24M"],
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Payment frequency").grid(row=4, column=0, sticky="w")
        self.freq_var = tk.StringVar(value="Semiannual")
        ttk.Combobox(inputs, textvariable=self.freq_var, values=["Annual", "Semiannual", "Quarterly"], width=12, state="readonly")\
            .grid(row=4, column=1, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Fixed rate (coupon)").grid(row=1, column=2, sticky="w")
        self.coupon_var = tk.StringVar(value="")
        self.coupon_entry = ttk.Entry(inputs, textvariable=self.coupon_var, width=12)
        self.coupon_entry.grid(row=1, column=3, sticky="w", padx=(0, 10))
        self.use_par = tk.BooleanVar(value=True)
        ttk.Checkbutton(inputs, text="Use par (auto)", variable=self.use_par, command=self._toggle_coupon)\
            .grid(row=1, column=4, sticky="w")
        self._toggle_coupon()

        # Market / discounting
        ttk.Label(inputs, text="Market / discounting", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w", pady=(0, 2))
        ttk.Label(inputs, text="Flat OIS (risk-free) rate").grid(row=2, column=2, sticky="w")
        self.riskfree_var = tk.StringVar(value="0.02")  # 2%
        ttk.Entry(inputs, textvariable=self.riskfree_var, width=12).grid(row=2, column=3, sticky="w", padx=(0, 10))

        # Rate model
        ttk.Label(inputs, text="Rate model (Vasicek 1-factor)", font=("Segoe UI", 10, "bold")).grid(row=0, column=5, sticky="w", pady=(0, 2))

        ttk.Label(inputs, text="Initial short rate r0").grid(row=1, column=5, sticky="w")
        self.shortrate_var = tk.StringVar(value="0.02")
        ttk.Entry(inputs, textvariable=self.shortrate_var, width=10).grid(row=1, column=6, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Mean reversion a").grid(row=2, column=5, sticky="w")
        self.meanrev_var = tk.StringVar(value="0.10")
        ttk.Entry(inputs, textvariable=self.meanrev_var, width=10).grid(row=2, column=6, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Long-run mean b").grid(row=3, column=5, sticky="w")
        self.longmean_var = tk.StringVar(value="0.02")
        ttk.Entry(inputs, textvariable=self.longmean_var, width=10).grid(row=3, column=6, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Volatility σ (abs)").grid(row=4, column=5, sticky="w")
        self.volatility_var = tk.StringVar(value="0.01")
        ttk.Entry(inputs, textvariable=self.volatility_var, width=10).grid(row=4, column=6, sticky="w", padx=(0, 10))

        # Credit & funding
        ttk.Label(inputs, text="Counterparty credit", font=("Segoe UI", 10, "bold")).grid(row=0, column=7, sticky="w", pady=(0, 2))
        ttk.Label(inputs, text="Recovery Rc").grid(row=1, column=7, sticky="w")
        self.countrecov_var = tk.StringVar(value="0.40")
        ttk.Entry(inputs, textvariable=self.countrecov_var, width=10).grid(row=1, column=8, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Hazard hc (flat)").grid(row=2, column=7, sticky="w")
        self.countdef_var = tk.StringVar(value="0.02")
        ttk.Entry(inputs, textvariable=self.countdef_var, width=10).grid(row=2, column=8, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Own credit (DVA)", font=("Segoe UI", 10, "bold")).grid(row=3, column=7, sticky="w", pady=(6, 2))
        ttk.Label(inputs, text="Recovery Rb").grid(row=4, column=7, sticky="w")
        self.ownrecov_var = tk.StringVar(value="0.40")
        ttk.Entry(inputs, textvariable=self.ownrecov_var, width=10).grid(row=4, column=8, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Hazard hb (flat)").grid(row=5, column=7, sticky="w")
        self.owndef_var = tk.StringVar(value="0.015")
        ttk.Entry(inputs, textvariable=self.owndef_var, width=10).grid(row=5, column=8, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Funding (FVA)", font=("Segoe UI", 10, "bold")).grid(row=6, column=7, sticky="w", pady=(6, 2))
        ttk.Label(inputs, text="Funding spread f − r").grid(row=7, column=7, sticky="w")
        self.fund_var = tk.StringVar(value="0.005")
        ttk.Entry(inputs, textvariable=self.fund_var, width=10).grid(row=7, column=8, sticky="w", padx=(0, 10))

        # Simulation
        ttk.Label(inputs, text="Simulation controls", font=("Segoe UI", 10, "bold")).grid(row=6, column=5, sticky="w", pady=(6, 2))
        ttk.Label(inputs, text="Paths").grid(row=7, column=5, sticky="w")
        self.paths_var = tk.StringVar(value="20000")
        ttk.Entry(inputs, textvariable=self.paths_var, width=10).grid(row=7, column=6, sticky="w", padx=(0, 10))

        ttk.Label(inputs, text="Time steps (grid)").grid(row=7, column=2, sticky="w")
        self.step_var = tk.StringVar(value="260")  # ~weekly for 5y
        ttk.Entry(inputs, textvariable=self.step_var, width=10).grid(row=7, column=3, sticky="w", padx=(0, 10))

        # Run button
        ttk.Button(inputs, text="Run Pricing", command=self.run_pricing).grid(row=8, column=0, sticky="w", pady=(10, 0))

        # ---------- Results ----------
        res = ttk.LabelFrame(main, text="Results", padding=10)
        res.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        for c in range(6):
            res.columnconfigure(c, weight=1)

        self.lbl_clean = ttk.Label(res, text="Clean price: —")
        self.lbl_clean.grid(row=0, column=0, sticky="w")
        self.lbl_cva = ttk.Label(res, text="CVA: —")
        self.lbl_cva.grid(row=0, column=1, sticky="w")
        self.lbl_dva = ttk.Label(res, text="DVA: —")
        self.lbl_dva.grid(row=0, column=2, sticky="w")
        self.lbl_fva = ttk.Label(res, text="FVA: —")
        self.lbl_fva.grid(row=0, column=3, sticky="w")
        self.lbl_xva = ttk.Label(res, text="Total xVA: —")
        self.lbl_xva.grid(row=0, column=4, sticky="w")
        self.lbl_allin = ttk.Label(res, text="All-in price: —", font=("Segoe UI", 10, "bold"))
        self.lbl_allin.grid(row=0, column=5, sticky="w")

        # ---------- Messages ----------
        io = ttk.LabelFrame(main, text="Assumptions & Messages", padding=10)
        io.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        io.columnconfigure(0, weight=1)
        io.rowconfigure(0, weight=1)

        self.assumptions_box = scrolledtext.ScrolledText(io, height=10, wrap="word")
        self.assumptions_box.grid(row=0, column=0, sticky="nsew")

        # ---------- Plot ----------
        plot = ttk.LabelFrame(main, text="Exposure profiles (EE / ENE)", padding=10)
        plot.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)

        self._figure = plt.Figure(figsize=(6, 2.5))
        ax = self._figure.add_subplot(111)
        ax.set_title("Run pricing to generate EE / ENE profiles")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Exposure")
        ax.grid(True, alpha=0.3)
        self._canvas = FigureCanvasTkAgg(self._figure, master=plot)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _toggle_coupon(self):
        if self.coupon_entry is None:
            return
        if self.use_par.get():
            self.coupon_entry.configure(state="disabled")
        else:
            self.coupon_entry.configure(state="normal")

    # =========================
    # Helpers
    # =========================
    @staticmethod
    def _parse_float(name, s):
        try:
            return float(str(s).strip())
        except Exception:
            raise ValueError(f"Invalid number for '{name}': {s}")

    @staticmethod
    def _parse_tenor(tenor_str: str) -> float:
        s = str(tenor_str).strip().upper()
        if s.endswith("Y"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 12.0
        return float(s)

    @staticmethod
    def _pay_freq_to_int(freq_label: str) -> int:
        m = {"Annual": 1, "Semiannual": 2, "Quarterly": 4}
        return m.get(freq_label, 2)

    @staticmethod
    def _par_swap_rate_flat(r, tenor_years, pay_freq):
        delta = 1.0 / pay_freq
        pay_times = np.arange(delta, tenor_years + 1e-12, delta)
        disc = np.exp(-r * pay_times)
        annuity = delta * np.sum(disc)
        p0T = np.exp(-r * tenor_years)
        return (1.0 - p0T) / annuity if annuity > 0 else 0.0

    # =========================
    # Core Monte Carlo engine
    # =========================
    @staticmethod
    def _simulate_vasicek(n_paths, n_steps, dt, r0, a, b, sigma):
        """
        Euler-Maruyama for Vasicek: dr = a(b-r)dt + sigma dW
        Returns r paths shape (n_paths, n_steps+1)
        """
        r = np.empty((n_paths, n_steps + 1), dtype=float)
        r[:, 0] = r0
        sqrt_dt = np.sqrt(dt)
        for k in range(n_steps):
            z = np.random.normal(size=n_paths)
            r[:, k + 1] = r[:, k] + a * (b - r[:, k]) * dt + sigma * z * sqrt_dt
        # clamp extremes to avoid numerical oddities
        np.clip(r, -0.05, 0.50, out=r)
        return r

    @staticmethod
    def _cumint_and_D0t(r, dt):
        """
        cum_int: ∫0^t r_s ds at each grid step (exclude the last point)
        D0t: exp(-cum_int)
        """
        cum_int = np.cumsum(r[:, :-1] * dt, axis=1)  # (paths, n_steps)
        D0t = np.exp(-cum_int)
        return cum_int, D0t

    @staticmethod
    def _idx_for_time(t, dt, n_steps):
        idx = int(round(t / dt)) - 1
        if idx < 0:
            idx = 0
        if idx > n_steps - 1:
            idx = n_steps - 1
        return idx

    def _swap_mtm_paths(self, cum_int, pay_idx, T_idx, dt, trade_sign):
        """
        Build payer-swap MTM along time grid using discount factors from cum_int.
        trade_sign: +1 for payer (receive float, pay fixed) ; -1 for receiver
        Returns mtm matrix shape (paths, n_steps)
        """
        n_paths, n_steps = cum_int.shape
        mtm = np.zeros((n_paths, n_steps), dtype=float)

        # For each valuation time k (grid), value remaining cashflows
        for k in range(n_steps):
            # stop if past last payment
            if k >= T_idx:
                break

            rem = pay_idx[pay_idx > k]
            if rem.size == 0:
                continue

            # Discounts from tk to ti: D(tk,ti) = exp( -(cum(ti) - cum(tk)) )
            Dtkti = np.exp(-(cum_int[:, rem] - cum_int[:, [k]]))  # (paths, #rem)

            # Fixed leg PV at tk = K * sum Δ * D(tk,ti) -- we'll scale by notional*K later
            fixed_leg = np.sum(Dtkti, axis=1)  # accrual Δ applied later as constant

            # Float leg at reset ~ 1 - P(tk, T)
            P_tkT = np.exp(-(cum_int[:, T_idx] - cum_int[:, k]))
            float_leg = 1.0 - P_tkT

            # Store raw legs; scale outside (need Δ and notional and coupon)
            # For now return (float_leg, fixed_leg vector) via mtm hack:
            mtm[:, k] = float_leg  # temporarily store float leg; fixed leg handled later via outer scale
            # We'll return both via separate arrays instead of overloading -> keep simple:
        return mtm  # not used directly; we compute legs inside run with full scaling

    # =========================
    # Run pricing
    # =========================
    def run_pricing(self):
        try:
            # ---------- Parse inputs ----------
            trade_type = self.trade_type.get()
            trade_sign = +1 if trade_type == "Payer" else -1
            notional = self._parse_float("Notional", self.notional_var.get())
            tenor_y = self._parse_tenor(self.tenor_var.get())
            pay_freq = self._pay_freq_to_int(self.freq_var.get())

            r_ois = self._parse_float("Flat OIS", self.riskfree_var.get())
            r0 = self._parse_float("Initial short rate r0", self.shortrate_var.get())
            a = self._parse_float("Mean reversion a", self.meanrev_var.get())
            b = self._parse_float("Long-run mean b", self.longmean_var.get())
            sigma = self._parse_float("Volatility σ", self.volatility_var.get())

            Rc = self._parse_float("Counterparty recovery Rc", self.countrecov_var.get())
            hc = self._parse_float("Counterparty hazard hc", self.countdef_var.get())
            Rb = self._parse_float("Own recovery Rb", self.ownrecov_var.get())
            hb = self._parse_float("Own hazard hb", self.owndef_var.get())
            f_minus_r = self._parse_float("Funding spread f-r", self.fund_var.get())

            n_paths = int(self._parse_float("Paths", self.paths_var.get()))
            n_steps = int(self._parse_float("Time steps", self.step_var.get()))
            if n_paths <= 0 or n_steps <= 0:
                raise ValueError("Paths and Time steps must be positive integers.")

            # Coupon: use par if requested or blank
            if self.use_par.get() or str(self.coupon_var.get()).strip() == "":
                coupon = self._par_swap_rate_flat(r_ois, tenor_y, pay_freq)
                used_par = True
            else:
                coupon = self._parse_float("Fixed rate (coupon)", self.coupon_var.get())
                used_par = False

            # ---------- Deterministic pre-compute ----------
            delta = 1.0 / pay_freq
            pay_times = np.arange(delta, tenor_y + 1e-12, delta)
            disc0 = np.exp(-r_ois * pay_times)
            annuity0 = delta * np.sum(disc0)
            P0T = np.exp(-r_ois * tenor_y)

            # Clean price at t=0 (payer = float - fixed)
            clean_price = notional * ( (1.0 - P0T) - coupon * annuity0 ) * trade_sign

            # ---------- Monte Carlo ----------
            dt = tenor_y / n_steps
            r = self._simulate_vasicek(n_paths, n_steps, dt, r0, a, b, sigma)
            cum_int, D0t = self._cumint_and_D0t(r, dt)  # (paths, n_steps)
            time_grid = (np.arange(1, n_steps + 1) * dt)

            # Indices for payment dates on the grid
            pay_idx = np.array([self._idx_for_time(t, dt, n_steps) for t in pay_times], dtype=int)
            T_idx = pay_idx[-1]

            # Pathwise MTM loop (efficiently)
            # We'll compute both legs inside the loop so we can scale correctly with Δ and coupon.
            mtm = np.zeros((n_paths, n_steps), dtype=float)

            for k in range(n_steps):
                if k >= T_idx:
                    break
                rem = pay_idx[pay_idx > k]
                if rem.size == 0:
                    continue

                # D(tk,ti)
                Dtkti = np.exp(-(cum_int[:, rem] - cum_int[:, [k]]))  # (paths, rem)
                fixed_leg = coupon * delta * np.sum(Dtkti, axis=1)    # scaled fixed leg
                P_tkT = np.exp(-(cum_int[:, T_idx] - cum_int[:, k]))
                float_leg = 1.0 - P_tkT
                payer_mtm = notional * (float_leg - fixed_leg)        # payer = receive float - pay fixed
                mtm[:, k] = payer_mtm * trade_sign                    # flip for receiver

            # ---------- Exposures ----------
            Eplus = np.maximum(mtm, 0.0)
            Eminus = np.maximum(-mtm, 0.0)

            # Undiscounted EE/ENE for plotting
            EE = np.mean(Eplus, axis=0)
            ENE = np.mean(Eminus, axis=0)

            # Discount to t=0 for integration
            EE_disc0 = np.mean(Eplus * D0t, axis=0)   # shape (n_steps,)
            ENE_disc0 = np.mean(Eminus * D0t, axis=0)
            E_disc0 = np.mean((Eplus - Eminus) * D0t, axis=0)  # signed expected discounted exposure

            # ---------- Default densities (flat hazards) ----------
            Sc = np.exp(-hc * time_grid)
            Sb = np.exp(-hb * time_grid)
            dPDc = hc * Sc * dt
            dPDb = hb * Sb * dt

            # ---------- xVA integrals ----------
            CVA = (1.0 - Rc) * np.sum(EE_disc0 * dPDc)
            DVA = (1.0 - Rb) * np.sum(ENE_disc0 * dPDb)
            FVA = np.sum(E_disc0 * f_minus_r * dt)

            xva_total = -CVA + DVA - FVA
            all_in = clean_price + xva_total

            # ---------- Update Results ----------
            self.lbl_clean.configure(text=f"Clean price: {clean_price:,.2f}")
            self.lbl_cva.configure(text=f"CVA: {CVA:,.2f}")
            self.lbl_dva.configure(text=f"DVA: {DVA:,.2f}")
            self.lbl_fva.configure(text=f"FVA: {FVA:,.2f}")
            self.lbl_xva.configure(text=f"Total xVA: {xva_total:,.2f}")
            self.lbl_allin.configure(text=f"All-in price: {all_in:,.2f}")

            # Assumptions / summary text
            txt = []
            txt.append("✅ Pricing completed.\n")
            txt.append("— Trade")
            txt.append(f"   • Type: {trade_type}")
            txt.append(f"   • Notional: {notional:,.0f}")
            txt.append(f"   • Tenor: {tenor_y:.2f}y   • Frequency: {pay_freq}x ({self.freq_var.get()})")
            txt.append(f"   • Coupon: {coupon:.6%} {'(par auto-derived)' if used_par else '(user)'}\n")
            txt.append("— Discounting @ flat OIS")
            txt.append(f"   • r = {r_ois:.4%}   • P(0,T) = {P0T:.6f}   • Annuity = {annuity0:.6f}")
            txt.append(f"   • Clean price = {clean_price:,.2f}\n")
            txt.append("— Rate model (Vasicek)")
            txt.append(f"   • r0={r0:.4%}, a={a:.3f}, b={b:.4%}, σ={sigma:.4%}, dt={dt:.5f}y")
            txt.append(f"   • Paths={n_paths:,}, Steps={n_steps:,}\n")
            txt.append("— Credit & Funding")
            txt.append(f"   • Counterparty: Rc={Rc:.0%}, hc={hc:.2%}")
            txt.append(f"   • Own: Rb={Rb:.0%}, hb={hb:.2%}")
            txt.append(f"   • Funding spread (f−r)={f_minus_r:.2%}\n")
            txt.append("— xVA results")
            txt.append(f"   • CVA={CVA:,.2f}, DVA={DVA:,.2f}, FVA={FVA:,.2f}")
            txt.append(f"   • Total xVA={xva_total:,.2f}, All-in={all_in:,.2f}\n")
            txt.append("Sanity: if hc=hb=0 and f−r=0 ⇒ xVA≈0; if coupon=par ⇒ clean≈0.")
            self.assumptions_box.delete("1.0", tk.END)
            self.assumptions_box.insert(tk.END, "\n".join(txt))

            # ---------- Plot exposures ----------
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.plot(time_grid, EE, label="EE (undiscounted)")
            ax.plot(time_grid, ENE, label="ENE (undiscounted)")
            ax.set_title("Expected Exposure (EE) and Expected Negative Exposure (ENE)")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Exposure")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            self._canvas.draw()

        except Exception as e:
            messagebox.showerror("Input error", str(e))


def main():
    root = tk.Tk()
    app = XvaPricingDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
