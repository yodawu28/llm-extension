import { execFileSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const repoRoot = path.resolve(__dirname, "..")
const packageJsonPath = path.join(repoRoot, "package.json")
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"))

const extensionName = String(packageJson.name || "extension")
const extensionVersion = String(packageJson.version || "0.0.0")
const buildDir = path.join(repoRoot, "build")
const prodDir = path.join(buildDir, "chrome-mv3-prod")
const artifactName = `${extensionName}-${extensionVersion}-chrome-mv3-prod.zip`
const artifactPath = path.join(buildDir, artifactName)
const plasmoArtifactPath = path.join(buildDir, "chrome-mv3-prod.zip")

if (!fs.existsSync(prodDir)) {
  throw new Error(`Missing production build directory: ${prodDir}`)
}

fs.rmSync(artifactPath, { force: true })
fs.rmSync(plasmoArtifactPath, { force: true })

execFileSync("/usr/bin/zip", ["-qr", artifactPath, "."], {
  cwd: prodDir,
  stdio: "inherit"
})

const stats = fs.statSync(artifactPath)

if (stats.size === 0) {
  throw new Error(`Packaged zip is empty: ${artifactPath}`)
}

console.log(`Created ${path.relative(repoRoot, artifactPath)} (${stats.size} bytes)`)
