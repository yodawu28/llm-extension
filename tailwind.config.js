/** @type {import('tailwindcss').Config} */
module.exports = {
  mode: "jit",
  darkMode: "class",
  content: [
    "./*.{ts,tsx}",
    "./popup/**/*.{ts,tsx}",
    "./contents/**/*.{ts,tsx}",
    "./sidepanel/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  prefix: "plasmo-",
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1'
        }
      }
    }
  },
  plugins: []
}
