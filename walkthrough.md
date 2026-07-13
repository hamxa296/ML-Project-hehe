# Quick and Yummy — SEO/SSR Upgrade: What Was Done

**Completed:** July 8, 2026  
**Commit:** `f3a851f` on `main`

---

## ✅ What Was Completed

### Task 0 — Fixed Vercel Routing (BLOCKING — done first)
**File:** `frontend/vercel.json`

The old config had a single catch-all `/*` → `/index.html` that was swallowing every path — including `/robots.txt`, `/sitemap.xml`, and all recipe/category URLs — and returning the empty React shell to all crawlers.

The new config adds **path-specific rewrites before the catch-all**, proxying the following paths transparently to the Django backend at `recipe296.alwaysdata.net`:

| Path | Proxied to |
|---|---|
| `/robots.txt` | Django |
| `/sitemap.xml` | Django |
| `/recipes/` | Django SSR |
| `/recipes/<slug>` | Django SSR |
| `/categories/<slug>` | Django SSR |
| `/about` | Django SSR |
| `/contact` | Django SSR |
| `/privacy-policy` | Django SSR |
| `/*` (everything else) | `/index.html` (React SPA — unchanged) |

The SPA catch-all stays last so all authenticated/app routes continue to work exactly as before.

---

### Task 0d — Fixed `FRONTEND_URL` env var on Alwaysdata server (manual)
The production server's `.env` had `FRONTEND_URL=http://localhost:5174`, causing `robots.txt` to declare `Sitemap: http://localhost:5174/sitemap.xml`.

**Verified fixed** — `curl.exe -s https://recipe296.alwaysdata.net/robots.txt` now returns:
```
Sitemap: https://quickandyummy.com/sitemap.xml
```

---

### Task 1 — `django-vite` (intentionally skipped)
The upgrade guide originally suggested adding `django-vite` to load the React bundle from Django templates. This was **not needed** for the chosen architecture: the Django SSR layer serves standalone HTML pages for crawlers — it does not hydrate or mount the React app. The React SPA continues to be served normally by Vercel for all other paths.

---

### Task 2 — Base SSR Template
**File:** `backend/templates/base_public.html`

Created a shared base template that all public SSR pages extend. Includes:
- Full `<head>` with title, meta description, canonical URL, Open Graph tags, Twitter Card, and `robots: index, follow, max-image-preview:large`
- `{% block schema %}` slot for JSON-LD injection per page
- `{% block content %}` for page-specific visible HTML
- A minimal, self-contained inline CSS stylesheet (no external dependencies — the page is readable with zero JS or network requests)
- Consistent site header with nav links: Home, Recipes, Blog, About, Contact
- Footer with About, Contact, Privacy Policy, All Recipes links + Facebook social link

---

### Task 3 — Recipe Detail SSR (P0)
**Files:**
- `backend/apps/recipes/views.py` → `RecipeSSRDetailView`
- `backend/templates/recipes/recipe_detail.html`
- `backend/config/urls.py` → `path('recipes/<slug:slug>/', ...)`

The view:
- Fetches only `status='published'` recipes
- Builds a full **Recipe JSON-LD schema** dict in Python via `json.dumps` (the recommended safe approach — no template string interpolation) including: `name`, `url`, `datePublished`, `dateModified`, `description`, `author`, `totalTime`, `recipeCategory`, `recipeIngredient`, `recipeInstructions`, and `image` (resolved from R2 via `recipe.image.url`)
- Queries 3 related recipes from the same category for the "You might also like" block (server-rendered real links — meaningful internal linking)

The template renders: breadcrumb, hero image, title, byline, time/difficulty bar, description, ingredients list, instructions, tags, and related recipes grid. All with correct `itemprop` microdata attributes.

---

### Task 4 — Recipe Index & Category Hub Pages (P1)
**Files:**
- `backend/apps/recipes/views.py` → `RecipeSSRListView`, `CategorySSRDetailView`
- `backend/templates/recipes/recipe_list.html`
- `backend/templates/recipes/category_detail.html`
- `backend/config/urls.py` → `path('recipes/', ...)`, `path('categories/<slug:slug>/', ...)`

`RecipeSSRListView`:
- Paginates all published recipes (24 per page)
- Shows category filter links with recipe counts (e.g. "Dinner (12)")
- Every recipe card is a real `<a>` link — this builds the internal link graph that crawlers follow

`CategorySSRDetailView`:
- Renders a category hub page with `<h1>`, recipe count, optional description, full recipe grid, and cross-links to all other categories

**Acceptance criteria met:** every published recipe is reachable from the `/recipes/` hub via a crawlable `<a>` link. No orphan pages.

---

### Task 5 — Homepage SSR Shell (P0)
**Files:**
- `backend/apps/cms/views.py` → `HomepageSSRView`
- `backend/templates/home.html`
- `backend/config/urls.py` → `path('', ...)`

The homepage crawler shell includes:
- WebSite JSON-LD schema with `SearchAction` (sitelinks searchbox)
- `<h1>` with brand tagline
- Category hubs grid with recipe counts
- Featured recipes grid (up to 6)
- Recent recipes grid (up to 12)
- "View All Recipes" CTA

---

### Task 6 — About / Contact / Privacy Policy Pages (P1)
**Files:**
- `backend/apps/cms/views.py` → `AboutSSRView`, `ContactSSRView`, `PrivacyPolicySSRView`
- `backend/templates/pages/about.html`
- `backend/templates/pages/contact.html`
- `backend/templates/pages/privacy_policy.html`
- `backend/config/urls.py` → `about/`, `contact/`, `privacy-policy/`

**About page** — includes Person JSON-LD schema, author bio section with E-E-A-T signals, and a dynamic "What you'll find here" category grid. Directly addresses the author identity gap flagged in the audit.

**Contact page** — static HTML form (Formspree-compatible), social links. Required for AdSense/Raptive eligibility.

**Privacy policy** — covers data collection, cookies, third-party services (Firebase, Cloudflare R2, Neon), and user rights.

All three pages are linked from the footer of `base_public.html` so every SSR page links to them.

---

### Task 7 — `robots.txt` (already existed — verified clean)
The `robots_txt` view in `cms/views.py` was already correct (`Allow: /`, no blanket `Disallow`). The only fix needed was the `FRONTEND_URL` env var (Task 0d) so the `Sitemap:` line pointed at the right domain.

**Verified:** `curl.exe -s https://quickandyummy.com/robots.txt` returns the correct file from the Django backend (not the React SPA's `index.html`). ✅

---

### Task 8 — XML Sitemap (already existed — verified working)
The `sitemap_xml` view in `cms/views.py` was already built and generates a valid XML sitemap including homepage, blog index, all published recipes, all published posts, and all pages. With `FRONTEND_URL` now set correctly, all `<loc>` entries point to `https://quickandyummy.com/...`.

---

## ⚡ Live Verification Results

| Check | Result |
|---|---|
| `recipe296.alwaysdata.net/robots.txt` Sitemap line | ✅ `https://quickandyummy.com/sitemap.xml` |
| `recipe296.alwaysdata.net/recipes/` returns HTML | ✅ Full `<!DOCTYPE html>` with recipe cards |
| `quickandyummy.com/robots.txt` returns real content | ✅ (not React `index.html`) |

---

## 🕐 Still Pending (not code — manual steps)

### Task 9 — Google Search Console + Bing Webmaster Tools
No code required. You need to:
1. Go to [Google Search Console](https://search.google.com/search-console)
2. Add and verify `quickandyummy.com` (use DNS TXT record method — easiest)
3. Submit sitemap: `https://quickandyummy.com/sitemap.xml`
4. Use the **URL Inspection tool** on `https://quickandyummy.com/recipes/` to confirm Google sees the SSR HTML
5. Repeat for [Bing Webmaster Tools](https://www.bing.com/webmasters) (can import from GSC directly)

### Task 10 — Star ratings + related recipes engagement signals (P2 — do after above is live and stable)
- A `RecipeRating` model surfaced in the API and wired into the `aggregateRating` JSON-LD block
- The related-recipes block already exists as server-side links ✅ (done in Task 3)

---

## 📁 Files Changed Summary

| File | Change |
|---|---|
| `frontend/vercel.json` | Rewrites: specific paths proxy to Django before SPA catch-all |
| `frontend/public/robots.txt` | Added for Vercel static fallback |
| `backend/config/urls.py` | All SSR URL patterns added |
| `backend/apps/recipes/views.py` | `RecipeSSRDetailView`, `RecipeSSRListView`, `CategorySSRDetailView` added |
| `backend/apps/cms/views.py` | `HomepageSSRView`, `AboutSSRView`, `ContactSSRView`, `PrivacyPolicySSRView` added; `Q` import added |
| `backend/templates/base_public.html` | **NEW** — shared SEO head + nav + footer |
| `backend/templates/home.html` | **NEW** — homepage SSR shell |
| `backend/templates/recipes/recipe_detail.html` | **NEW** — recipe SSR page with JSON-LD |
| `backend/templates/recipes/recipe_list.html` | **NEW** — paginated recipe index |
| `backend/templates/recipes/category_detail.html` | **NEW** — category hub page |
| `backend/templates/pages/about.html` | **NEW** — author bio + E-E-A-T |
| `backend/templates/pages/contact.html` | **NEW** — contact form |
| `backend/templates/pages/privacy_policy.html` | **NEW** — privacy policy |
