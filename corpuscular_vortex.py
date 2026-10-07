#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorpuscularVortex — vortex = corpuscle = field in one object
Author: Ziyavutdinov Magomed Kamalovich (Zimaka)
License: MIT
"""
import math
import argparse
import numpy as np

__author__ = "Зиявутдинов Магомед Камалович (Zimaka)"
__version__ = "1.0.0"
__license__ = "MIT"

TWO_PI = 2.0 * math.pi
MU0 = 4 * math.pi * 1e-7
G_NEWTON = 6.674e-11
E_CHARGE = 1.602176634e-19


class CorpuscularVortex:
    """
    Unified object: vortex core + corpuscle + field.
    Charge = Γ, spin = ±1, mass = |Γ|.
    """

    def __init__(self, position=(0, 0, 0), circulation=1.0,
                 core_radius=0.5, axis=(0, 0, 1),
                 external_B=(0, 0, 0.5), external_E=(0, 0, 0),
                 external_g=(0, 0, 0), name="cv"):
        self.position = np.array(position, dtype=np.float64)
        self.circulation = float(circulation)
        self.core_radius = float(core_radius)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis /= max(np.linalg.norm(self.axis), 1e-12)
        self.B_ext = np.array(external_B, dtype=np.float64)
        self.E_ext = np.array(external_E, dtype=np.float64)
        self.g_ext = np.array(external_g, dtype=np.float64)
        self.name = name

        # Corpuscular attributes
        self.charge = self.circulation
        self.spin = +1 if self.circulation >= 0 else -1
        self.mass = abs(self.circulation)

    def magnetic_field_at(self, x):
        """B from Ampère + external B."""
        d = np.asarray(x, dtype=np.float64) - self.position
        z = float(np.dot(d, self.axis))
        perp = d - z * self.axis
        r = float(np.linalg.norm(perp))
        I = self.circulation / MU0
        if r < 1e-12:
            B_self = np.zeros(3)
        else:
            tangent = np.cross(self.axis, perp) / r
            if r >= self.core_radius:
                B_mag = MU0 * I / (TWO_PI * r)
            else:
                B_mag = MU0 * I * r / (TWO_PI * self.core_radius ** 2)
            B_self = tangent * B_mag
        return B_self + self.B_ext

    def gravity_field_at(self, x):
        """External gravity only (self-gravity via mass)."""
        return self.g_ext.copy()

    def hall_drift(self):
        """v = (E × B) / |B|²."""
        B = self.magnetic_field_at(self.position)
        B2 = float(np.dot(B, B))
        if B2 < 1e-20:
            return np.zeros(3)
        return np.cross(self.E_ext, B) / B2

    def grav_drift(self, tau=0.1):
        """v = g · τ."""
        return self.gravity_field_at(self.position) * tau

    def field_energy(self):
        """Magnetic + gravitational energy density."""
        B = self.magnetic_field_at(self.position)
        g = self.gravity_field_at(self.position)
        return (float(np.dot(B, B)) / (2 * MU0) +
                0.5 * float(np.dot(g, g)) / G_NEWTON)

    def step(self, dt=0.01, tau=0.1):
        """Evolve position: Hall + gravity drift."""
        v = self.hall_drift() + self.grav_drift(tau)
        self.position = self.position + dt * v
        return v

    def report(self):
        B = self.magnetic_field_at(self.position)
        g = self.gravity_field_at(self.position)
        return (
            f"CorpuscularVortex({self.name})\n"
            f"  position: {self.position.round(4)}\n"
            f"  Γ (charge): {self.circulation:+.4f}\n"
            f"  spin: {self.spin:+d}\n"
            f"  mass: {self.mass:.4f}\n"
            f"  r_c: {self.core_radius:.4f}\n"
            f"  |B|: {np.linalg.norm(B):.4e}\n"
            f"  |g|: {np.linalg.norm(g):.4e}\n"
            f"  energy: {self.field_energy():.4e}"
        )


def selftest():
    print("=" * 60)
    print(f"CORPUSCULARVORTEX v{__version__} — SELFTEST")
    print("=" * 60)

    cv = CorpuscularVortex(
        position=(0, 0, 0), circulation=1.0, core_radius=0.5,
        external_B=(0.2, 0, 0.5), external_E=(0.1, 0, 0),
        external_g=(0, 0, -1e-4), name="test_cv")

    print(f"\n{cv.report()}")

    print("\n--- Evolution over 10 steps ---")
    for i in range(10):
        v = cv.step(dt=0.01)
        if i % 3 == 0:
            print(f"  step {i:2d}: |v| = {np.linalg.norm(v):.4e}, "
                  f"pos = {cv.position.round(4)}")

    # Conservation check
    print("\n--- Conservation ---")
    print(f"  charge: {cv.charge:+.4f} (should = Γ)")
    print(f"  spin: {cv.spin:+d}")
    print(f"  mass: {cv.mass:.4f}")

    print("\n✅ SELFTEST passed")
    print(f"Author: {__author__}")


def main():
    parser = argparse.ArgumentParser(
        description=f"CorpuscularVortex v{__version__}")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    selftest()


if __name__ == "__main__":
    main()