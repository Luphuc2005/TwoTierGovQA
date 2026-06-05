/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
            },
            colors: {
                /* ── Core palette ────────────────────── */
                primary: {
                    50:  '#EEF2FF',
                    100: '#DBEAFE',
                    200: '#BFDBFE',
                    300: '#93C5FD',
                    400: '#60A5FA',
                    500: '#3B82F6',
                    600: '#2563EB',
                    700: '#1D4ED8',
                    800: '#1E40AF',
                    900: '#1E3A8A',
                },
                secondary: {
                    500: '#0F766E',
                    600: '#0D9488',
                },
                /* ── Sidebar (light) ─────────────────── */
                sidebar: {
                    DEFAULT: '#FFFFFF',
                    hover:   '#F1F5F9',
                    active:  '#EAF2FF',
                    border:  '#E2E8F0',
                },
                /* ── Layout backgrounds ──────────────── */
                surface: {
                    DEFAULT: '#F5F7FB',
                    card:    '#FFFFFF',
                    input:   '#FFFFFF',
                },
                /* ── Text ────────────────────────────── */
                txt: {
                    primary:   '#1E293B',
                    secondary: '#475569',
                    muted:     '#94A3B8',
                    label:     '#64748B',
                    inverse:   '#F8FAFC',
                },
                /* ── Chat bubbles ────────────────────── */
                bubble: {
                    user: '#2563EB',
                    ai:   '#FFFFFF',
                    aiBg: '#EEF6FF',
                },
                /* ── Borders & misc ──────────────────── */
                bdr: {
                    DEFAULT: '#E2E8F0',
                    light:   '#F1F5F9',
                    medium:  '#CBD5E1',
                },
            },
            boxShadow: {
                'card':    '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
                'card-lg': '0 4px 16px rgba(0,0,0,0.08)',
                'card-xl': '0 8px 24px rgba(0,0,0,0.10)',
                'bubble':  '0 1px 4px rgba(0,0,0,0.06)',
                'input':   '0 1px 4px rgba(0,0,0,0.05)',
                'sidebar': '1px 0 0 #E2E8F0',
                'header':  '0 1px 3px rgba(0,0,0,0.04)',
            },
            animation: {
                "bounce-dot": "bounce-dot 1.4s infinite ease-in-out both",
                "fade-in":    "fadeIn 0.3s ease-out",
                "slide-up":   "slideUp 0.4s ease-out",
                "pulse-soft": "pulseSoft 2s infinite ease-in-out",
            },
            keyframes: {
                "bounce-dot": {
                    "0%, 80%, 100%": { transform: "scale(0)" },
                    "40%":          { transform: "scale(1)" },
                },
                fadeIn: {
                    from: { opacity: "0", transform: "translateY(8px)" },
                    to:   { opacity: "1", transform: "translateY(0)" },
                },
                slideUp: {
                    from: { opacity: "0", transform: "translateY(16px)" },
                    to:   { opacity: "1", transform: "translateY(0)" },
                },
                pulseSoft: {
                    "0%, 100%": { opacity: "0.4" },
                    "50%":     { opacity: "1" },
                },
            },
        },
    },
    plugins: [],
};
