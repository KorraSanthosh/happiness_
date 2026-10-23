# Demonstration video — pending

**Status:** not recorded yet. No video link exists; do not mark it complete until the final recording is uploaded.

## Where to insert the link

Replace the line under **"Demonstration video"** in the README (section *Team*):

```
**Demonstration video:** Pending recording by the team leader. ...
```

with:

```
**Demonstration video:** <final video URL> (recorded by Korra Santhosh, <date>)
```

## Recording checklist

- [ ] Rehearse once with a timer using `docs/GlucoTwin_20_Minute_Speaking_Script.pdf` (target about 20 minutes).
- [ ] Start the backend (`cd backend && python -m uvicorn app.main:app --port 8000`) and frontend (`cd frontend && npm run dev`); open `http://localhost:5173`.
- [ ] Settings → Theme: Light; Data mode: Real patients; Alert mode: Standard.
- [ ] Check that **Patient #2000** loads and that "Example early warning" shows Day 5 · 21:30.
- [ ] Close notifications and other apps; record at 1920×1080 with the microphone tested.
- [ ] Show slides from `docs/GlucoTwin_Presentation.pdf` exactly as the script says.
- [ ] Say clearly that the demo uses anonymised research recordings replayed as if live (and that synthetic mode uses virtual patients), and that this is an experimental prototype, not clinically validated.
- [ ] Do not upload personal health data during the recording.
- [ ] Upload the video, set the visibility the challenge requires, and paste the link into the README.
