export interface UserSettings {
  mindReaderScanMode: "auto" | "manual" | "off"
  mindReaderPopupsEnabled: boolean
}

export const USER_SETTINGS_STORAGE_KEY = "userSettings"

export const DEFAULT_USER_SETTINGS: UserSettings = {
  mindReaderScanMode: "auto",
  mindReaderPopupsEnabled: true
}

export async function readUserSettings(): Promise<UserSettings> {
  const result = await chrome.storage.local.get(USER_SETTINGS_STORAGE_KEY)
  const stored = result[USER_SETTINGS_STORAGE_KEY] || {}
  const legacyAutoScanEnabled = stored.mindReaderAutoScanEnabled
  const storedScanMode = stored.mindReaderScanMode

  let mindReaderScanMode: UserSettings["mindReaderScanMode"] = DEFAULT_USER_SETTINGS.mindReaderScanMode

  if (storedScanMode === "auto" || storedScanMode === "manual" || storedScanMode === "off") {
    mindReaderScanMode = storedScanMode
  } else if (legacyAutoScanEnabled === false) {
    mindReaderScanMode = "off"
  }

  return {
    mindReaderScanMode,
    mindReaderPopupsEnabled:
      typeof stored.mindReaderPopupsEnabled === "boolean"
        ? stored.mindReaderPopupsEnabled
        : DEFAULT_USER_SETTINGS.mindReaderPopupsEnabled
  }
}

export async function updateUserSettings(
  patch: Partial<UserSettings>
): Promise<UserSettings> {
  const nextSettings = {
    ...(await readUserSettings()),
    ...patch
  }

  await chrome.storage.local.set({
    [USER_SETTINGS_STORAGE_KEY]: nextSettings
  })

  return nextSettings
}
