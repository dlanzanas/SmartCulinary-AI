# SmartCulinary AI - UI/UX Quick Reference

## 🎨 Design System Essentials

**Colors:**
- Primary: `#10B981` (Fresh Green)
- Text: `#1F2937` (Charcoal)
- Background: `#FFFFFF` (White)
- Cards: `#F3F4F6` (Light Gray)

**Typography:** Inter font
- Titles: 32px/600
- Headers: 24px/600
- Body: 16px/400

**Spacing:** 8px base unit

---

## 📱 Three Core Screens

### 1. Scanner (Recognition Module)
**Flow:** Camera → Capture → CNN Prediction → Verification

**Key Elements:**
- Live camera preview with 3x3 grid
- "Scan Item" button (green, rounded)
- Top-5 predictions with confidence bars
- Flash and gallery quick actions

**Success Criteria:**
- < 200ms inference latency
- > 85% top-1 accuracy
- Real-world lighting tolerance

---

### 2. Virtual Pantry (Inventory Module)
**Flow:** Verify Item → Edit if needed → Add to Pantry

**Key Elements:**
- Item verification screen (Correct/Wrong/Edit)
- Pantry list with emoji icons
- Swipe-to-delete gestures
- "Find Recipes" CTA button

**Features:**
- Manual name editing
- Category organization
- Empty state messaging

---

### 3. Recipe Recommendations (Recipe Engine)
**Flow:** Query API → Filter by Pantry → Display Ranked Results

**Key Elements:**
- Recipe cards with images
- Match indicator (e.g., "Uses 4/5 items")
- Rating + time + difficulty
- Detailed view with instructions

**API Integration:**
- Spoonacular or Edamam
- Filter: Time, difficulty, cuisine
- Highlight missing ingredients

---

## ✨ Key Animations

| Interaction | Duration | Effect |
|-------------|----------|--------|
| Camera shutter | 200ms | White flash + haptic |
| Result reveal | 500ms | Slide up from bottom |
| Confidence bar | 800ms | Fill animation |
| Add to pantry | 300ms | Checkmark + toast |
| Page transition | 300ms | Slide + fade |

---

## 🏗️ 3-Tier Architecture

**Tier 1 (Flutter):**
- ScannerScreen
- PantryScreen
- RecipesScreen

**Tier 2 (Flask/Node.js):**
- `/api/scan` - ML inference
- `/api/pantry` - CRUD operations
- `/api/recipes` - External API proxy

**Tier 3 (Data):**
- Firebase/PostgreSQL (user data)
- TFLite model (3.17 MB)
- Spoonacular API (recipes)

---

## ✅ Implementation Checklist

- [ ] Set up Flutter project (Android + Web)
- [ ] Create design system constants
- [ ] Build camera screen with TFLite integration
- [ ] Build verification + pantry screens
- [ ] Set up backend API (Flask/Node.js)
- [ ] Integrate recipe API (Spoonacular)
- [ ] Add animations and polish
- [ ] Test on Android device
- [ ] Deploy web version
- [ ] Measure success metrics

---

## 🎯 MVP Success Metrics

| Metric | Target |
|--------|--------|
| Scan-to-Recipe Time | < 30 seconds |
| Top-1 Accuracy | > 85% |
| Top-5 Accuracy | > 95% |
| Inference Latency | < 200ms |
| Model Size | < 15 MB ✅ (3.17 MB) |

---

**Next Step:** Begin Flutter project setup and component library development.
