---
name: SlotMemory
description: Object-Centric KV Memory for Streaming Long-Video Generation
colors:
  primary: "#2563eb"
  primary-hover: "#1d4ed8"
  accent: "#0ea5e9"
  secondary: "#64748b"
  text-primary: "#1e293b"
  text-secondary: "#64748b"
  text-light: "#94a3b8"
  background-primary: "#ffffff"
  background-secondary: "#f8fafc"
  background-accent: "#f1f5f9"
  border-color: "#e2e8f0"
typography:
  display:
    fontFamily: "'Inter', sans-serif"
    fontWeight: 800
    lineHeight: 1.1
  title:
    fontFamily: "'Inter', sans-serif"
    fontWeight: 700
  body:
    fontFamily: "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "16px"
    lineHeight: 1.6
rounded:
  md: "12px"
  lg: "16px"
spacing:
  base: "1.5rem"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.background-primary}"
    rounded: "{rounded.md}"
---

# Design System: SlotMemory

## 1. Overview

**Creative North Star: "The Dynamic Innovator"**

This visual system brings academic rigor to life through dynamic energy and bright, memorable presentation. It embraces the "Grand but Grounded" principle, balancing an impressive, engaging layout with clear scientific communication. This system explicitly rejects monotonous, generic academic templates and overly drab color schemes, aiming instead to leave a lasting visual memory that encourages adoption and citation.

**Key Characteristics:**
- Bright, energetic color palette anchored by a vibrant blue.
- Bold, highly legible typography using the Inter font family.
- Generous use of white space and layered shadows to create depth.
- Smooth transitions and interactive polish that don't overwhelm the content.

## 2. Colors

A vibrant, academic palette designed to stand out while maintaining a rigorous, authoritative feel.

### Primary
- **Vibrant Academic Blue** (#2563eb): The core brand color, used for primary actions, prominent links, and accents. It conveys energy and innovation.

### Secondary
- **Slate Secondary** (#64748b): Used for supporting text and elements that require a more subdued presence alongside the primary blue.

### Tertiary
- **Cyan Accent** (#0ea5e9): Provides a striking contrast to the primary blue, often used in gradients or secondary highlights.

### Neutral
- **Primary Background** (#ffffff): The clean canvas for reading dense academic content.
- **Secondary Background** (#f8fafc): Used for alternating sections (like hero blocks) to create rhythm.
- **Primary Text** (#1e293b): Near-black for maximum readability on light backgrounds.
- **Border** (#e2e8f0): Subtle division between structural elements.

**The Brighter Future Rule.** Avoid overly dark or drab color schemes. The design must leverage its bright blues and clean whites to feel modern and "alive."

## 3. Typography

**Display Font:** Inter (with sans-serif fallback)
**Body Font:** Inter (with Segoe UI, -apple-system fallback)

**Character:** Clean, highly legible, and optimized for screen reading. The heavy weights in display sizes provide a grand, confident feel.

### Hierarchy
- **Display** (800, 2.5rem+, 1.1): The main publication title and hero statements. Engineered for maximum impact.
- **Title** (700, 1.5rem+): Section headers. Often accompanied by a bottom accent border or stylistic underline.
- **Body** (400, 16px, 1.6): Main paragraphs and explanatory text. Spaced comfortably for reading complex academic concepts.

**The Bold Presentation Rule.** Use deliberate scale and weight contrast (like 800-weight display vs. 400-weight body) to create a confident hierarchy.

## 4. Elevation

The system embraces a **Layered & Lifted** philosophy. Shadows are used extensively to create a clear sense of depth, hierarchy, and interaction.

### Shadow Vocabulary
- **Small** (`box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05)`): Subtle lift, often used for code blocks or minor elements.
- **Medium** (`box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), ...`): The default rest state for interactive elements like cards and some buttons.
- **Large** (`box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), ...`): Used on hover states for images and buttons to indicate interactivity.
- **Extra Large** (`box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), ...`): Reserved for prominent elements like video banners or dropdown menus.

**The Interactive Lift Rule.** Interactive elements should visually lift (using `transform: translateY` and increased shadow) when hovered, providing satisfying feedback.

## 5. Components

Components feel **Polished & Restrained**, balancing academic necessity with modern interactive polish.

### Buttons
- **Shape:** Rounded (12px radius).
- **Primary:** Dark background (`#1e293b`) that shifts to Primary Blue (`#2563eb`) on hover, creating a striking reveal.
- **Hover / Focus:** Translates up by 2px with an increased shadow (md to lg).

### Cards / Media Containers
- **Corner Style:** Large radius (16px) for a soft, friendly boundary.
- **Background:** White (`#ffffff`) or transparent, often with a subtle border (`#e2e8f0`).
- **Shadow Strategy:** Medium shadow at rest, Large shadow and an upward translation on hover.

### Navigation / Dropdowns
- **Style:** Pill-shaped triggers with cleanly bordered dropdown panels.
- **Interaction:** The dropdown utilizes the Extra Large shadow and smooth opacity/transform transitions for a polished reveal.

## 6. Do's and Don'ts

### Do:
- **Do** use the Vibrant Academic Blue (`#2563eb`) to draw attention to key actions like "Paper" or "Code" links.
- **Do** employ layered shadows to give depth to images, videos, and interactive cards.
- **Do** maintain a clean, white/light-gray background to ensure maximum legibility for academic text.

### Don't:
- **Don't** use monotonous, generic academic templates (e.g., causvid.github.io).
- **Don't** create default "cookie-cutter" paper pages that fail to leave a lasting visual memory.
- **Don't** use overly dark or drab color schemes.