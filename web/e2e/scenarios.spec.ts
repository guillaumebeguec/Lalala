import { expect, test } from "@playwright/test";

// Parcours complets en mode mock (FRONT_PLAN.md F3) : Capture → Searching → Result.
test("memory: text description to result, not this one, no confident match", async ({ page }) => {
  await page.addInitScript(() => { (window as unknown as { __LALALA_FAST__: boolean }).__LALALA_FAST__ = true; });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("You know… that one song?");
  await page.getByLabel("Describe the song").fill("whistling, car ad, ~2008");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page).toHaveURL(/\/result\//);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Whistle Song");
  await expect(page.getByText("Best match · high confidence")).toBeVisible();
  await expect(page.getByText("Why this match")).toBeVisible();
  for (const next of ["Open Road", "Sunny Side", "Night Drive", "We couldn’t pin it down."]) {
    await page.getByRole("button", { name: "Not this one" }).click();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(next);
  }
  await page.getByRole("button", { name: "Add a detail" }).click();
  await expect(page.getByLabel("Describe the song")).toHaveValue("whistling, car ad, ~2008");
});

test("replay query param plays a scene end to end with steps", async ({ page }) => {
  await page.goto("/?replay=cover");
  await expect(page.getByText("Finding your song…")).toBeVisible();
  await expect(page.getByText("Isolating the music")).toBeVisible();
  await expect(page).toHaveURL(/\/result\//, { timeout: 20_000 });
  await expect(page.getByText("Best match · medium confidence")).toBeVisible();
  await expect(page.getByText("Your recording")).toBeVisible();
  await page.getByRole("button", { name: "View score details" }).click();
  await expect(page.getByRole("dialog", { name: "Score details" })).toBeVisible();
});

test("noisy scene shows the noise badge and a promoted candidate", async ({ page }) => {
  await page.goto("/?replay=noisy");
  await expect(page.getByText("Background noise detected")).toBeVisible();
  await expect(page).toHaveURL(/\/result\//, { timeout: 20_000 });
  await page.getByRole("button", { name: /Night Drive/ }).first().click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Night Drive");
});

test("humming scene shows the humming input kind and melody proof", async ({ page }) => {
  await page.goto("/?replay=humming");
  await expect(page.getByText("Humming · 8 s")).toBeVisible();
  await expect(page.getByText("Melody matching")).toBeVisible();
  await expect(page).toHaveURL(/\/result\//, { timeout: 20_000 });
  await expect(page.getByText("Your hummed melody matches the vocal line.")).toBeVisible();
});
