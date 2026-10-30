import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "../src/App";
import { copy } from "../src/copy";

describe("Capture screen", () => {
  it("shows the headline, listen button and brief bar", () => {
    render(<MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(copy.capture.h1);
    expect(screen.getByRole("button", { name: copy.capture.listen })).toBeInTheDocument();
    expect(screen.getByLabelText(copy.capture.describe)).toHaveAttribute("placeholder", copy.capture.placeholder);
    expect(screen.getByRole("button", { name: copy.capture.search })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: copy.capture.addVideo })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: copy.capture.addAudio })).toBeInTheDocument();
  });
});
