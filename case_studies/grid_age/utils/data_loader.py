"""Data loader for GridAges time-series data.

Loads real-world profiles for:
- Electricity prices
- Solar PV availability
- Wind availability
- Load demands
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class DataLoader:
    """Loads and manages time-series data for microgrid simulation.

    Data structure (from data.pkl):
        data = {
            'train': {
                'price': {location: array(8760,)},
                'solar': {zone: array(8760,)},
                'wind': {zone: array(8760,)},
                'load': {bus: array(8760,)}
            },
            'test': {same structure}
        }

    8760 hours = 365 days × 24 hours
    """

    def __init__(self, data_path: Optional[Path] = None, split: str = 'train'):
        """Initialize data loader.

        Args:
            data_path: Path to data.pkl file (default: auto-detect)
            split: 'train' or 'test'
        """
        if data_path is None:
            # Auto-detect path relative to this file
            current_dir = Path(__file__).parent.parent
            data_path = current_dir / 'data.pkl'

        self.data_path = Path(data_path)
        self.split = split

        # Load data
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        if split not in data:
            raise ValueError(f"Split '{split}' not found in data. Available: {list(data.keys())}")

        self.data = data[split]

        # Store available keys
        self.price_locations = list(self.data.get('price', {}).keys())
        self.solar_zones = list(self.data.get('solar', {}).keys())
        self.wind_zones = list(self.data.get('wind', {}).keys())
        self.load_buses = list(self.data.get('load', {}).keys())

        print(f"Loaded {split} data:")
        print(f"  - Price locations: {len(self.price_locations)}")
        print(f"  - Solar zones: {len(self.solar_zones)}")
        print(f"  - Wind zones: {len(self.wind_zones)}")
        print(f"  - Load buses: {len(self.load_buses)}")

    def get_price_profile(
        self,
        location: Optional[str] = None,
        start_hour: int = 0,
        num_hours: int = 24
    ) -> np.ndarray:
        """Get electricity price profile.

        Args:
            location: Price location key (uses first if None)
            start_hour: Starting hour (0-8759)
            num_hours: Number of hours to extract

        Returns:
            Price array of shape (num_hours,) in $/MWh
        """
        if location is None:
            location = self.price_locations[0] if self.price_locations else None

        if location is None or location not in self.data['price']:
            # Return default price profile if not found
            return np.full(num_hours, 50.0)

        prices = self.data['price'][location]
        end_hour = start_hour + num_hours

        # Handle wraparound
        if end_hour <= len(prices):
            return prices[start_hour:end_hour]
        else:
            # Wrap around to beginning
            part1 = prices[start_hour:]
            part2 = prices[:end_hour - len(prices)]
            return np.concatenate([part1, part2])

    def get_solar_profile(
        self,
        zone: Optional[str] = None,
        start_hour: int = 0,
        num_hours: int = 24
    ) -> np.ndarray:
        """Get solar PV availability profile.

        Args:
            zone: Solar zone key (uses first if None)
            start_hour: Starting hour (0-8759)
            num_hours: Number of hours to extract

        Returns:
            Availability array of shape (num_hours,) in [0, 1]
        """
        if zone is None:
            zone = self.solar_zones[0] if self.solar_zones else None

        if zone is None or zone not in self.data['solar']:
            # Return default solar profile (daytime pattern)
            hours = np.arange(num_hours) % 24
            return np.where((hours >= 6) & (hours < 18),
                            np.sin(np.pi * (hours - 6) / 12),
                            0.0)

        solar = self.data['solar'][zone]
        end_hour = start_hour + num_hours

        if end_hour <= len(solar):
            return solar[start_hour:end_hour]
        else:
            part1 = solar[start_hour:]
            part2 = solar[:end_hour - len(solar)]
            return np.concatenate([part1, part2])

    def get_wind_profile(
        self,
        zone: Optional[str] = None,
        start_hour: int = 0,
        num_hours: int = 24
    ) -> np.ndarray:
        """Get wind availability profile.

        Args:
            zone: Wind zone key (uses first if None)
            start_hour: Starting hour (0-8759)
            num_hours: Number of hours to extract

        Returns:
            Availability array of shape (num_hours,) in [0, 1]
        """
        if zone is None:
            zone = self.wind_zones[0] if self.wind_zones else None

        if zone is None or zone not in self.data['wind']:
            # Return default wind profile
            return np.clip(
                0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, num_hours)),
                0.0, 1.0
            )

        wind = self.data['wind'][zone]
        end_hour = start_hour + num_hours

        if end_hour <= len(wind):
            return wind[start_hour:end_hour]
        else:
            part1 = wind[start_hour:]
            part2 = wind[:end_hour - len(wind)]
            return np.concatenate([part1, part2])

    def get_load_profile(
        self,
        bus: Optional[str] = None,
        start_hour: int = 0,
        num_hours: int = 24
    ) -> np.ndarray:
        """Get load demand profile.

        Args:
            bus: Load bus key (uses first if None)
            start_hour: Starting hour (0-8759)
            num_hours: Number of hours to extract

        Returns:
            Load array of shape (num_hours,) in MW
        """
        if bus is None:
            bus = self.load_buses[0] if self.load_buses else None

        if bus is None or bus not in self.data['load']:
            # Return default load profile
            return np.full(num_hours, 0.2)

        loads = self.data['load'][bus]
        end_hour = start_hour + num_hours

        if end_hour <= len(loads):
            return loads[start_hour:end_hour]
        else:
            part1 = loads[start_hour:]
            part2 = loads[:end_hour - len(loads)]
            return np.concatenate([part1, part2])

    def get_episode_data(
        self,
        episode: int = 0,
        episode_length: int = 24,
        price_location: Optional[str] = None,
        solar_zone: Optional[str] = None,
        wind_zone: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Get all profiles for a complete episode.

        Args:
            episode: Episode number (determines starting hour)
            episode_length: Episode length in hours
            price_location: Price location key
            solar_zone: Solar zone key
            wind_zone: Wind zone key

        Returns:
            Dict with 'price', 'solar', 'wind' arrays
        """
        # Calculate starting hour (cycle through year)
        start_hour = (episode * episode_length) % 8760

        return {
            'price': self.get_price_profile(price_location, start_hour, episode_length),
            'solar': self.get_solar_profile(solar_zone, start_hour, episode_length),
            'wind': self.get_wind_profile(wind_zone, start_hour, episode_length),
        }


# Singleton instance
_loader_instance = None


def get_data_loader(split: str = 'train') -> DataLoader:
    """Get singleton data loader instance.

    Args:
        split: 'train' or 'test'

    Returns:
        DataLoader instance
    """
    global _loader_instance
    if _loader_instance is None or _loader_instance.split != split:
        _loader_instance = DataLoader(split=split)
    return _loader_instance
