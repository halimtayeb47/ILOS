# Security

ILOS is a grid simulation, not a physical vehicle controller. It does not model
braking distance, sensor error, vehicle dimensions or continuous motion.

The modern API defaults to localhost in the documented launch command. Set
`ILOS_API_KEY` and use an authenticated reverse proxy before exposing it beyond a
trusted local network. Requests cannot choose checkpoint paths. CBS requests have
bounded map size, fleet size, search horizon, node count and runtime.

Load only model files you created or trust. The recovered PyTorch loader uses
`weights_only=True`; PPO ZIP metadata uses serialization that is not safe for
untrusted uploads. File hashes detect accidental changes, not a malicious author.
The archived edge API is retained for research reproduction and is not hardened
for internet exposure.

Report vulnerabilities privately to the repository maintainers through their
GitHub profiles. Do not include tokens or private data in public issues.
