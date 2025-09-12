const NEWS = require('./index.js');

describe('NEWS: Nobody Ever Wins Sh*T', () => {
  test('should return correct version', () => {
    expect(NEWS.version).toBe('1.0.0');
  });

  test('should have the right motto', () => {
    expect(NEWS.motto).toBe('Nobody Ever Wins Sh*T... but we keep trying anyway!');
  });

  test('should return project status', () => {
    const status = NEWS.getStatus();
    
    expect(status).toHaveProperty('version');
    expect(status).toHaveProperty('motto');
    expect(status).toHaveProperty('lastUpdated');
    expect(status).toHaveProperty('features');
    expect(status).toHaveProperty('automations');
    
    expect(Array.isArray(status.features)).toBe(true);
    expect(Array.isArray(status.automations)).toBe(true);
    expect(status.features.length).toBeGreaterThan(0);
    expect(status.automations.length).toBeGreaterThan(0);
  });

  test('should include expected features', () => {
    const status = NEWS.getStatus();
    
    expect(status.features).toContain('Continuous Integration');
    expect(status.features).toContain('Automated Dependency Updates');
    expect(status.features).toContain('Security Scanning');
  });

  test('should include expected automations', () => {
    const status = NEWS.getStatus();
    
    expect(status.automations).toContain('GitHub Actions CI/CD');
    expect(status.automations).toContain('Dependabot Updates');
    expect(status.automations).toContain('Security Alerts');
  });

  test('should never be winning (because nobody ever wins)', () => {
    expect(NEWS.areWeWinning()).toBe(false);
  });

  test('should have a valid lastUpdated timestamp', () => {
    const status = NEWS.getStatus();
    const lastUpdated = new Date(status.lastUpdated);
    
    expect(lastUpdated).toBeInstanceOf(Date);
    expect(lastUpdated.getTime()).not.toBeNaN();
  });

  test('display method should not throw', () => {
    // Mock console.log to avoid cluttering test output
    const originalLog = console.log;
    console.log = jest.fn();
    
    expect(() => NEWS.display()).not.toThrow();
    
    // Restore console.log
    console.log = originalLog;
  });
});