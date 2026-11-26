"""Shared upper-left force panel HTML, independent of simulator and hardware readers."""

from html import escape


def force_overlay(
    live: float | None,
    completed: float | None,
    status: str,
    *,
    sides: tuple[float | None, float | None] | None = None,
) -> str:
    """Render the original single-arm panel or live left/right fingertip normal loads.

    Args:
        live: Live normal-load sum in N, or unavailable.
        completed: Latched native completion measurement for the single-arm panel.
        status: Plain-text availability or simulation timestamp.
        sides: Optional left/right sums; replaces the native-completion section.
    """
    live_text = "—" if live is None else f"{live:.2f} N"
    completed_text = "—" if completed is None else f"{completed:.2f} N"
    title = "Live contact ΣFn" if sides is None else "Total fingertip ΣFn"
    details = (
        '<div style="margin-top:14px;font-size:15px;font-weight:600">Grasp complete ΣFn</div>'
        f'<div style="font-size:34px;font-weight:700;color:#087b51">{completed_text}</div>'
        '<div style="font-size:11px;color:#58616b">Native completion · saved value</div>'
    )
    if sides is not None:
        details = "".join(
            f'<div style="margin-top:8px;font-size:18px">{side} ΣFn: '
            + ("—" if value is None else f"{value:.2f} N")
            + "</div>"
            for side, value in zip(("Left", "Right"), sides)
        )
    return (
        '<div id="contact-force-overlay" style="position:fixed;top:16px;left:16px;z-index:1000;'
        "pointer-events:none;background:rgba(255,255,255,.96);color:#17212b;padding:16px 22px;"
        'border-radius:10px;box-shadow:0 2px 14px #0002;min-width:240px;font-family:sans-serif">'
        f'<div style="font-size:15px;font-weight:600">{title}</div>'
        f'<div style="font-size:34px;font-weight:700;color:#c45b00">{live_text}</div>'
        f'<div style="font-size:11px;color:#58616b">{escape(status)}</div>' + details + "</div>"
    )
