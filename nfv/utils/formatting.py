import torch

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def str_beautify(s):
    return str(s).replace("_", " ").title()


def format_scientific(value, sig_digits=2, latex=False):
    """Formats a value in scientific notation with the specified number of significant digits."""
    try:
        base, exponent = "{:.{}e}".format(value, sig_digits - 1).split("e")
    except ValueError:
        breakpoint()
    exponent = int(exponent)  # convert to integer to remove extra zero
    if latex:
        return f"{base}\\mathrm{{e}}^{{{exponent}}}"
    else:
        return f"{base}e{str(exponent).translate(SUPERSCRIPT)}"


def format(values, latex=False, bold_val=None):
    if latex:
        if bold_val is not None and abs(torch.mean(values) - bold_val) < 1e-8:
            return f"\\(\\bm{{{format_scientific(torch.mean(values), latex=True)}}}\\)\\std{{\\({format_scientific(torch.std(values), sig_digits=1, latex=True)}\\)}}".replace(
                "-", "\\shortminus "
            )
        else:
            return f"\\({format_scientific(torch.mean(values), latex=True)}\\)\\std{{\\({format_scientific(torch.std(values), sig_digits=1, latex=True)}\\)}}".replace(
                "-", "\\shortminus "
            )
    else:
        return f"{format_scientific(torch.mean(values))} (± {format_scientific(torch.std(values), sig_digits=1)})"
