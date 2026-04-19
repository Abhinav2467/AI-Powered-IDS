# 🎤 MAHE Mobility Challenge - Demo Script

**Estimated Run Time:** 3 Minutes
**Presenter:** [Your Name / Team]

---

### Step 1: The Hook (0:00 - 0:30)

*(Start clearly projecting to the judges without looking at your screen)*

**Speaker:** "Welcome to our prototype for CAN-Guard AI. We are tackling the core vulnerability found in modern vehicle networks: Lateral movement. When a threat actor bypasses an air-gapped Infotainment system or OTA connection, they can jump to the Gateway ECU and start spoofing life-critical commands on the internal CAN bus—like sending unauthorized *Full Brake* requests to CAN ID `0x200`. Here's our real-time defense."

### Step 2: The Attack Dashboard (0:30 - 1:30)

*(Launch `python3 04_main_integration.py` or switch to the already running Tkinter Dashboard tab "Overview" & "Charts")*

**Speaker:** "In our simulation, we injected random fuzzing and rapid-fire brake actuations. If you look at our live Edge AI telemetry..." 
*(Point to the histograms and scatter plot)* 
"...you can see a distinct, immediate separation. The Isolation Forest algorithm evaluates payload length and packet timing. Normal telemetry sits tightly clustered, but anomalous requests dive significantly into the negative decision matrix. Because this relies purely on an unsupervised IF algorithm—this process operates safely at scale on lightweight Edge processing nodes and operates within average gateway stack latency."

### Step 3: Mitigation vs Shutdown (1:30 - 2:00)

*(Click on the "Incidents" or "Safety pie" Chart)*

**Speaker:** "Crucially, we've developed a true **Fail-Operational** mechanism—not just Fail-Safe. Rather than paralyzing the entire Braking ECU, we specifically isolate the attack vectors. The system automatically shifts to alert-only for low-confidence deviations, but immediately institutes cryptographic CMAC signing validations (and blocking operations) the second high-confidence anomalies surge—as seen here in the alert log outputs."

### Step 4: The LLM Cyber-Analyst (2:00 - 3:00)

*(Click on the final "Insights" tab. Optionally type a question into the text box if the API is operating)*

**Speaker:** "And finally, because manual packet analysis is impossible during an intrusion—we incorporated an LLM Cyber-Analyst. Connected securely to the local pipeline, it decodes and maps the exact security posture, confirming we maintained 'Fail-Operational' status across this test."

*(Conclude)*
**Speaker:** "Measurable precision. Zero reliance on the cloud. Quantum-ready lightweight encryption readiness. That's CAN-Guard. We're happy to answer any questions." 
