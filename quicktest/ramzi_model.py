import numpy as np


def ring_block(theta, phi_bias, phi_rt, tau):
    """Single resonant sub-block transfer function.

    Parameters
    ----------
    theta : float
        Internal phase parameter of the resonant block.
    phi_bias : float
        Static phase bias.
    phi_rt : np.ndarray
        Frequency-dependent round-trip phase.
    tau : float
        Amplitude transmission factor for one round trip.
    """
    num = (np.exp(1j * theta) - 1.0) / 2.0 - tau * np.exp(1j * (phi_bias + theta + phi_rt))
    den = 1.0 - tau * (1.0 - np.exp(1j * theta)) * np.exp(1j * (phi_rt + phi_bias)) / 2.0
    return num / den



def simulate_ramzi(
    f=None,
    E1=0.0 + 0.0j,
    E2=1.0 + 0.0j,
    Ki=0.5,
    Ko=0.5,
    thetai=0.5 * np.pi,
    thetao=0.5 * np.pi,
    fait=0.495 * np.pi,
    faib=-0.495 * np.pi,
    fai1=-0.0468 * np.pi,
    fai2=-0.6842 * np.pi,
    fai3=-0.0518 * np.pi,
    fai4=-0.6198 * np.pi,
    theta1=-0.622 * np.pi,
    theta2=-0.73 * np.pi,
    theta3=-0.622 * np.pi,
    theta4=-0.73 * np.pi,
    Alfadb=15.0,
    ng=4.3,
    L1=350e-6,
    L2=3000e-6,
    L3=350e-6,
    L4=3000e-6,
    c=3e8,
):
    """Simulate the RAMZI-like spectral response from the MATLAB script."""
    if f is None:
        f = np.arange(1.93523e14, 1.93558e14 + 0.5 * 0.00000025e14, 0.00000025e14)

    alfadb = Alfadb / 2.0
    alfa = alfadb * np.log(10.0) / 10.0

    tau1 = np.exp(-alfa * L1)
    tau2 = np.exp(-alfa * L2)
    tau3 = np.exp(-alfa * L3)
    tau4 = np.exp(-alfa * L4)

    phi1 = 2.0 * np.pi * f * L1 * ng / c
    phi2 = 2.0 * np.pi * f * L2 * ng / c
    phi3 = 2.0 * np.pi * f * L3 * ng / c
    phi4 = 2.0 * np.pi * f * L4 * ng / c

    # First MZI stage
    E3 = np.sqrt(1.0 - Ki) * E1 - 1j * np.sqrt(Ki) * E2
    E4 = -1j * np.sqrt(Ki) * E1 + np.sqrt(1.0 - Ki) * E2
    E5 = E3
    E6 = np.exp(1j * thetai) * E4
    E7 = np.sqrt(1.0 - Ki) * E5 - 1j * np.sqrt(Ki) * E6
    E8 = -1j * np.sqrt(Ki) * E5 + np.sqrt(1.0 - Ki) * E6

    # Two arms, each cascaded with two resonant blocks
    c1 = ring_block(theta1, fai1, phi1, tau1)
    c2 = ring_block(theta2, fai2, phi2, tau2)
    c3 = ring_block(theta3, fai3, phi3, tau3)
    c4 = ring_block(theta4, fai4, phi4, tau4)

    A1 = np.exp(1j * fait) * c1 * c2
    A2 = np.exp(1j * faib) * c3 * c4

    E9 = A1 * E7
    E10 = A2 * E8

    # Second MZI stage
    E11 = np.sqrt(1.0 - Ko) * E9 - 1j * np.sqrt(Ko) * E10
    E12 = -1j * np.sqrt(Ko) * E9 + np.sqrt(1.0 - Ko) * E10
    E13 = E11
    E14 = np.exp(1j * thetao) * E12
    E15 = np.sqrt(1.0 - Ko) * E13 - 1j * np.sqrt(Ko) * E14
    E16 = -1j * np.sqrt(Ko) * E13 + np.sqrt(1.0 - Ko) * E14

    C1 = np.abs(E15) ** 2
    C2 = np.abs(E16) ** 2
    CdB1 = 10.0 * np.log10(C1)
    CdB2 = 10.0 * np.log10(C2)
    lam = c / f

    return {
        "f": f,
        "lambda": lam,
        "E15": E15,
        "E16": E16,
        "C1": C1,
        "C2": C2,
        "CdB1": CdB1,
        "CdB2": CdB2,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
    }


if __name__ == "__main__":
    result = simulate_ramzi()
    print("lambda range (nm):", result["lambda"].min() * 1e9, result["lambda"].max() * 1e9)
    print("CdB2 range (dB):", result["CdB2"].min(), result["CdB2"].max())
